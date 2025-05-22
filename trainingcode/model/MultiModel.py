import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
from os.path import dirname as opd

sys.path.append(opd(__file__))

from encoder.heatmap_encoder import HeatmapEncoder
import math

class MultiModel(nn.Module):
    def __init__(self, input_dim, max_len, n_classes, graph, args, device=None):
        super(MultiModel, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = n_classes
        self.encoder_args = args.enc_args
        self.decoder_args = args.dec_args
        self.encoder_args.input_dim = input_dim
        self.encoder_args.max_len = max_len
        self.encoder_args.num_classes = n_classes
        self.encoder_args.concept_dim = args.concept_dim
        self.encoder_args.task_type = args.task_type
        self.decoder_args.input_dim = input_dim
        self.decoder_args.max_len = max_len
        self.decoder_args.n_classes = n_classes
        self.decoder_args.concept_dim = args.concept_dim

        self.pred_step = self.decoder_args.pred_step
        self.task_type = args.task_type
        self.kernel_size = args.kernel_size
        self.enc_type = args.enc_args.trimmer_types

        # Shared encoder
        self.encoder = self.build_encoder(self.encoder_args, graph)

        # Decoder for prediction
        self.decoder = self.build_decoder(self.decoder_args)

    def build_encoder(self, encoder_args, graph):
        return HeatmapEncoder(encoder_args, graph, device=self.device)

    def build_decoder(self, decoder_args):
        if self.task_type == 'classify':
            from decoder.PosAttentionDecoder import SingleClassDecoder
            return SingleClassDecoder(decoder_args)
        elif self.task_type == 'forcast':
            from decoder.DLinear import MultiDecoder
            return MultiDecoder(decoder_args)

    def train_refer_model(self, train_loader, val_loader, writter, save_path=None):
        if save_path == None:
            save_path == writter.log_dir
        self.encoder.train_refer_model(train_loader, val_loader, writter, save_path=save_path)

    def mask_attention(self, subenc, mask):
        def expand_mask(mask, t):

            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
    
            # Expand mask to [bs, seg, 1, n]
            mask = mask.unsqueeze(2)  # [bs, seg, n] -> [bs, seg, 1, n]
    
            # Create a tensor of ones with shape [bs, seg, t, n]
            ones = torch.ones(mask.size(0), mask.size(1), t, mask.size(3), device=mask.device)
    
            # Create a tensor of zeros with shape [bs, seg, t, n]
            zeros = torch.zeros(mask.size(0), mask.size(1), t, mask.size(3), device=mask.device)
    
            # Use the mask to select between ones and zeros
            expanded_mask = torch.where(mask, ones, zeros)
    
            return expanded_mask            

        if subenc.ndim == 3:
            subenc = subenc.unsqueeze(-1)
        bs, seg, t, n = subenc.size()
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1) 
        mask = expand_mask(mask, t)  
        mask = mask.to(subenc.device)
        high_attention_enc = subenc * mask
        low_attention_enc = subenc * (1 - mask)
        return high_attention_enc, low_attention_enc
    
    def forward(self, x, times, interp=False, threshold=0.7):
        x, means, stdev = self.normalize(x)
        x = x.to(self.device)
        times = times.to(self.device)
        subenc, all_seq, mask, segment_info = self.encoder(x, times)
        if torch.isnan(subenc).any():
            print("NaN values found in subenc")
        decoded, attention = self.decoder(subenc, interp=True)
        if torch.isnan(decoded).any():
            print("NaN values found in decoded")
        output = {}
        if len(self.enc_type) == 0:
            output['high_attention_encoding'] = subenc.unsqueeze(-1).repeat(1, 1, 1, self.encoder_args.input_dim)
            output['low_attention_encoding'] = None
        else:
            high_attention_enc, low_attention_enc = self.mask_attention(subenc, mask)
            output['high_attention_encoding'] = high_attention_enc
            output['low_attention_encoding'] = low_attention_enc
        output['all_seq'] = all_seq
        output['mask'] = mask
        if self.task_type == 'classify':
            output['high_attention_output'] = self.classify(decoded)
        elif self.task_type == 'forcast':
            output['high_attention_output'] = self.forecast(decoded, means, stdev)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        l1_loss = sum(torch.sum(torch.abs(param)) for name, param in self.named_parameters() if 'bias' not in name)
        l2_loss = sum(torch.sum(param.pow(2)) for name, param in self.named_parameters() if 'bias' not in name)
        regularization_loss = l1_loss + l2_loss

        output['regularization_loss'] = regularization_loss
        output['seg_info'] = segment_info
        if interp:
            mix_attention = self.load_attention_mask(mask, segment_info).to(x.device)
            if isinstance(attention, list):
                attention = torch.stack(attention, dim=-2)
            mix_attention = attention.mean(dim=-1) + mix_attention
            # subenc, mix_attention = self.threshold_decode(subenc, threshold, mix_attention)
            decoded, _ = self.decoder(subenc, interp=True)
            output['mix_attention'] = mix_attention
            output['decoder_encoding'] = attention
            if self.task_type == 'classify':
                output['high_attention_output'] = self.classify(decoded)
            elif self.task_type == 'forcast':
                output['high_attention_output'] = self.forecast(decoded, means, stdev)
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
            return output
        else:
            return output

    def classify(self, decoded):
        return decoded

    def forecast(self, decoded, means, stdev):
        denormalized = self.denormalize(decoded, means, stdev, self.pred_step)
        return denormalized

    def normalize(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev


        return x_enc, means, stdev

    def denormalize(self, x, means, stdev, length):
        denorm_x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, length, 1))
        denorm_x = denorm_x + (means[:, 0, :].unsqueeze(1).repeat(1, length, 1))

        return denorm_x
    
    def load_attention_mask(self, mask, seg_info):
        attention_mask = torch.zeros(mask.shape, dtype=torch.float32)
        for i, seg in enumerate(seg_info):
            if len(seg) > 10:
                for j, seg_j in enumerate(seg):
                    bs = seg_j['batch_id']
                    seg_id = seg_j['seg_id']
                    attention_mask[bs,seg_id,i] = seg_j['attention_mean']
            else:
                bs = seg['batch_id']
                seg_id = seg['seg_id']
                atten = seg['attention_mean']
                attention_mask[bs, seg_id] = atten
        return attention_mask

    def threshold_decode(self, x, threshold=None, mix_attention=None):
        """
        Decode the input tensor using the threshold.
        """

        def threshold_top_percent(mask: torch.Tensor, threshold: float) -> torch.Tensor:
            bs, t, n = mask.shape
            total_elements = t * n
            if threshold <= 0:
                return torch.zeros_like(mask)
            elif threshold >= 100:
                return torch.ones_like(mask)

            k = math.ceil(threshold * total_elements)
            k = min(max(k, 1), total_elements) 
            mask_flat = mask.view(bs, -1)
            topk_values, _ = torch.topk(mask_flat, k, dim=1)
            thresholds = topk_values[:, -1].unsqueeze(-1) 

            binary_mask = (mask_flat >= thresholds).float()
            return binary_mask.view_as(mask)
        if threshold is None or mix_attention is None:
            return x
        if len(x.shape) == 3 and len(mix_attention.shape) == 2:
            mix_attention = mix_attention.unsqueeze(2)
        mask = threshold_top_percent(mix_attention, threshold)

        x = x * mask
        return x, mask

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        self.load_state_dict(torch.load(model_path))

    def get_explanation(self, x, times, y):
        """
        Get the explanation for the model's prediction.
        Output: mask with the same shape as input x.
        """
        def map_mask_to_sequence(mask, seg_info, seq_len):

            # Initialize the output mask with zeros
            bs = mask.size(0)  # Batch size
            mapped_mask = torch.zeros(bs, seq_len, device=mask.device)
            mapped_list = []
            j = 0
            # Iterate over the segment information
            for seg in seg_info:
                if isinstance(seg, list):
                    mapped_mask_seg = torch.zeros(bs, seq_len, device=mask.device)
                    for seg_j in seg:
                        batch_id = seg_j['batch_id']
                        seg_id = seg_j['seg_id']
                        start = seg_j['start']
                        end = seg_j['end']

                        # Assign the mask value for the corresponding segment
                        mapped_mask_seg[batch_id, start:end] = mask[batch_id, seg_id, j]
                    mapped_list.append(mapped_mask_seg)
                    j += 1
                else:
                    batch_id = seg['batch_id']
                    seg_id = seg['seg_id']
                    start = seg['start']
                    end = seg['end']

                    # Assign the mask value for the corresponding segment
                    mapped_mask[batch_id, start:end] = mask[batch_id, seg_id]
            if len(mapped_list) > 0:
                mapped_mask = torch.stack(mapped_list, dim=-1)
            return mapped_mask
        # Ensure the model is on the correct device
        self.to(self.device)

        # Normalize the input
        x, means, stdev = self.normalize(x)
        x = x.to(self.device)
        times = times.to(self.device)
        y = y.to(self.device) if y is not None else None

        x = x.permute(1, 0, 2)
        times = times.permute(1, 0)

        # Pass through the encoder to get subencodings, mask, and segment info
        output = self.forward(x, times, interp=True)
        encoder_attention_mask = self.load_attention_mask(output['mask'], output['seg_info'])
        decoder_attention_mask = output['decoder_encoding'].mean(dim=-1).to(encoder_attention_mask.device)
        # Combine the attention masks
        combined_attention_mask = encoder_attention_mask + decoder_attention_mask
        attention = map_mask_to_sequence(combined_attention_mask, output['seg_info'], x.shape[1])
        # expland to [bs, t, n]
        if len(attention.shape) == 2:
            attention = attention.unsqueeze(-1).repeat(1, 1, self.encoder_args.input_dim)
        else:
            attention = attention.permute(1, 0, 2)
        return attention.permute(1, 0, 2)
    
def main():
    class Args:
        concept_dim = 32
        task_type = 'classify'
        kernel_size = 25
        num_classes = 10

        enc_args = type('', (), {})()
        enc_args.d_model = 512
        enc_args.max_len = 100
        enc_args.desired_threshold = 0.7
        enc_args.d_pe = 1
        enc_args.trimmer_types = ['sequence']
        enc_args.input_dim = 8
        enc_args.hidden_size = 256
        enc_args.num_layers = 2
        enc_args.encoder_type = 'lstm'
        enc_args.n_layers = 2
        enc_args.pred_step = 1
        enc_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dec_args = type('', (), {})()
        dec_args.input_dim = 128
        dec_args.hidden_dims = [64, 32]
        dec_args.pred_step = 96

    args = Args()
    input_dim = 20
    max_len = 50
    n_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiModel(input_dim, max_len, n_classes, args, device)
    model.to(device)

    # Create a dummy input tensor
    x = torch.randn(32, 10, input_dim).to(device)
    times = torch.randn(32, 10).to(device)

    # Forward pass
    output = model(x, times)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
