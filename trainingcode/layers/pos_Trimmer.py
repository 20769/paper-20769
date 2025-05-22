import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from os.path import join as opj
from os.path import dirname as opd
import math

sys.path.append(opd(__file__))
from refermodel import ReferModel

#PositionalEncodingTF 类
class PositionalEncodingTF(nn.Module):
    def __init__(self, d_pe, max_len):
        super(PositionalEncodingTF, self).__init__()
        self.d_pe = d_pe
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, d_pe)

    def forward(self, x):
        bs, t, d_model = x.shape
        pos_ids = torch.arange(t, device=x.device).unsqueeze(0).expand(bs, -1)
        pos_emb = self.pe(pos_ids)  # [bs, t, d_pe]
        return x + pos_emb

class TimeEncoder(nn.Module):
    def __init__(self, d_pe: int, max_len: int = 512):

        super().__init__()
        self.d_pe = d_pe
        
        pe = torch.zeros(max_len, d_pe)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_pe, 2).float() * (-math.log(10000.0) / d_pe)
        )
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1), :] 
        return x
    
class SequenceTrimmer(nn.Module):
    def __init__(self, model: nn.Module, args):
        super(SequenceTrimmer, self).__init__()
        self.model = model
        self.d_model = args.d_model
        self.max_len = args.max_len
        self.desired_threshold = args.desired_threshold
        self.positional_encoding = TimeEncoder(args.d_pe, args.max_len)

        self.fixed_max_segments = args.fixed_max_segments  # e.g. 8
        self.fixed_max_len = args.fixed_max_len           # e.g. 64        
        self.trimmer_types = args.trimmer_types
        self._init_transformers(args)
        self.cache = {}


    def _init_transformers(self, args):
        if 'Decomposition' in self.trimmer_types:
            from WaveletTransformer import WaveletTransform
            from SeriesDecomposition import SeriesDecomposition
            self.decomposition = SeriesDecomposition()
            self.wavelet_transformer = WaveletTransform(self.fixed_max_len, wavelet='auto', level=None, mode='symmetric')
        elif 'Wavelet' in self.trimmer_types:
            from WaveletTransformer import WaveletTransform
            self.wavelet_transformer = WaveletTransform(self.fixed_max_len, wavelet='auto', level=None, mode='symmetric')

    def init_refer_model(self, args):
        self.model = ReferModel(
            args.input_dim, args.hidden_size, args.num_layers, 
            args.task_type, args.num_classes, args.pred_step
        )

    def generate_heatmap(self, seq: torch.Tensor) -> torch.Tensor:


        self.model.eval()
        out, attention = self.model(seq)
        return attention

    def calculate_threshold(self, heatmap: torch.Tensor) -> torch.Tensor:

        k = int(heatmap.shape[1] * heatmap.shape[2] * (1 - self.desired_threshold))
        return torch.kthvalue(heatmap.view(heatmap.shape[0], -1), k, dim=1)[0]

    def apply_position_encoding(self, seq: torch.Tensor) -> torch.Tensor:

        bs, t, n = seq.shape
        seq_encoded = seq + self.positional_encoding(seq)
        return seq_encoded

    def frequency_transform(self, input_tensor: torch.Tensor, t_prime=None) -> torch.Tensor:

        transformed_tensor = torch.fft.fft(input_tensor, dim=1)
    
        if t_prime is None:
            t_prime = self.fixed_max_len
        truncated_tensor = transformed_tensor[:, :t_prime, :]
        
        return truncated_tensor

    def forward(self, seq: torch.Tensor, times: torch.Tensor):


        device = seq.device
        self.to(device)
        seq = self.apply_position_encoding(seq)

        heatmap = self.generate_heatmap(seq)

        high_attention_masks = heatmap

        if not self.trimmer_types:
            return self._handle_no_trimmer_types(seq)
        if seq.shape[-1] == 1:
            return self._process_trimmer_types(seq, high_attention_masks, times)
        else:
            segment_info = []
            all_output = []
            mask = []
            max_len = 0
            for i in range(seq.shape[-1]):
                seq_i = seq[:, :, i].unsqueeze(-1)
                high_attention_masks_i = high_attention_masks[:, :, i].unsqueeze(-1)
                segment_info_i, all_output_i, mask_i = self._process_trimmer_types(seq_i, high_attention_masks_i, times)
                segment_info.append(segment_info_i)
                all_output.append(all_output_i)
                mask.append(mask_i.unsqueeze(-1))
                if all_output_i.shape[2] > max_len:
                    max_len = all_output_i.shape[1]
            all_output = torch.cat(all_output, dim=-1)
            all_output = F.pad(all_output, (0, 0, 0, max_len - all_output.shape[1]), 'constant', 0)
            mask = torch.cat(mask, dim=-1)
            return segment_info, all_output, mask

    def _handle_no_trimmer_types(self, seq):

        seq_expanded = seq.unsqueeze(1)
        mask = torch.ones(size=(seq.shape[0], 1))
        # to bool
        mask = mask.bool()
        return None, seq_expanded, mask

    def _process_trimmer_types(self, seq, high_attention_masks, times):
        if 'Decomposition' in self.trimmer_types:
            return self._handle_decomposition(seq, high_attention_masks, times)
        else:
            return self._handle_other_types(seq, high_attention_masks, times)

    def _handle_decomposition(self, seq, high_attention_masks, times):

        trend, _ = self.decomposition(seq)
        (
            segment_info,
            all_output,
            mask
        ) = self._split_sequences_vectorized(seq, high_attention_masks)


        all_output = self._dynamic_padding_4d(all_output)
        wavelet_seq = self.wavelet_transformer.transform(trend, max_len=all_output.shape[2])
        wavelet_seq_tensor = torch.tensor(wavelet_seq, device=seq.device, dtype=seq.dtype)

        wavelet_seq_tensor = wavelet_seq_tensor.unsqueeze(1)  # [bs, 1, t', d_pe]
        freq_seq_tensor = self.frequency_transform(seq, t_prime=wavelet_seq_tensor.shape[2])
        freq_seq_tensor = freq_seq_tensor.unsqueeze(1)  # [bs, 1, t', d_pe]
        wavelet_seq_cat = torch.zeros_like(all_output)
        wavelet_seq_cat[:, :, :wavelet_seq_tensor.shape[2]] = wavelet_seq_tensor
        freq_seq_cat = torch.zeros_like(all_output)
        freq_seq_cat[:, :, :freq_seq_tensor.shape[2]] = freq_seq_tensor
        all_output = torch.cat([all_output, wavelet_seq_cat[:, :1, :, :]], dim=1)
        all_output = torch.cat([all_output, freq_seq_cat[:, :1, :, :]], dim=1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :2])], dim=1)

        return (segment_info, all_output, mask)
    
    def _handle_other_types(self, seq, high_attention_masks, times):

        (
            segment_info,
            all_output,
            mask
        ) = self._split_sequences_vectorized(seq, high_attention_masks)

        # 动态 padding
        all_output = self._dynamic_padding_4d(all_output)

        return (segment_info, all_output, mask)

    def _split_sequences_vectorized(self, seq, high_attention_masks):

        bs, t, d_pe = seq.shape
        device = seq.device

        if high_attention_masks.ndim == 3 and high_attention_masks.shape[-1] != 1:
            high_mask_reduced = high_attention_masks.any(dim=-1, keepdim=True)  # [bs, t, 1]
        else:
            high_mask_reduced = high_attention_masks

        pooling_mask = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        high_mask_reduced = pooling_mask(high_mask_reduced.permute(0, 2, 1)).permute(0, 2, 1)
        attention_mean = torch.mean(high_mask_reduced, dim=(1, 2))  # [bs]

        mask_changes = torch.diff(
            high_mask_reduced.float(),
            dim=1,
            prepend=torch.zeros(bs, 1, 1, device=device)
        )
        mean_change = torch.mean(torch.abs(mask_changes))
        std_change = torch.std(torch.abs(mask_changes))
        change_threshold = mean_change + std_change

        change_points = torch.where(torch.abs(mask_changes) > change_threshold / 1000)

        all_output = torch.zeros(
            bs, self.fixed_max_segments, self.fixed_max_len, d_pe, device=device
        )
        mask = torch.zeros(bs, self.fixed_max_segments, dtype=torch.bool, device=device)
        segment_info = []

        max_segments = 0
        for b in range(bs):
            idx_in_this_batch = change_points[1][change_points[0] == b]
            boundary_points = torch.cat([
                torch.tensor([0], device=device),
                idx_in_this_batch,
                torch.tensor([t], device=device)
            ]).unique(sorted=True)

            internal_boundaries = boundary_points[1:-1] if len(boundary_points) > 2 else []
            boundary_strengths = torch.abs(mask_changes[b, internal_boundaries, 0]) if len(internal_boundaries) > 0 else torch.tensor([], device=device)

            while (len(boundary_points) - 1) > self.fixed_max_segments and len(internal_boundaries) > 0:
                min_strength, min_idx = torch.min(boundary_strengths, dim=0)
                boundary_to_remove = internal_boundaries[min_idx]
                boundary_points = boundary_points[boundary_points != boundary_to_remove]
                internal_boundaries = boundary_points[1:-1] if len(boundary_points) > 2 else []
                if len(internal_boundaries) > 0:
                    boundary_strengths = torch.abs(mask_changes[b, internal_boundaries, 0])
                else:
                    break

            final_segments = []
            for i in range(len(boundary_points) - 1):
                start = boundary_points[i].item()
                end = boundary_points[i + 1].item()
                final_segments.append((start, end))
            for seg_idx, (start, end) in enumerate(final_segments):
                if seg_idx >= self.fixed_max_segments:
                    break
                curr_len = end - start
                if curr_len == 0:
                    continue
                if curr_len > self.fixed_max_len:
                    curr_len = self.fixed_max_len
                    end = start + curr_len

                segment_data = seq[b, start:end]
                seg_mask = high_mask_reduced[b, start:end]

                all_output[b, seg_idx, :curr_len] = segment_data
                mean_attention = seg_mask.mean()

                mask[b, seg_idx] = mean_attention > (attention_mean[b] * 1.5) 


                segment_info.append({
                    'batch_id': b,
                    'seg_id': seg_idx,
                    'start': start,
                    'end': end,
                    'attention_mean': mean_attention.item(),
                })
            if len(final_segments) > max_segments:
                max_segments = len(final_segments)

        all_output = all_output[:, :max_segments]
        mask = mask[:, :max_segments]
        return segment_info, all_output, mask

    def _dynamic_padding_4d(self, seq_4d: torch.Tensor):
        non_zero_time = (seq_4d != 0).any(dim=-1).any(dim=1)
        max_t = non_zero_time.sum(dim=1).max().item()

        non_zero_seg = (seq_4d != 0).any(dim=-1).any(dim=2)
        max_seg = non_zero_seg.sum(dim=1).max().item()
    
        seq_4d = seq_4d[:, :max_seg, :max_t, :]
    
        return seq_4d

    def _dynamic_padding_3d(self, seq_3d: torch.Tensor):

        bs, t, d_pe = seq_3d.shape
        return seq_3d