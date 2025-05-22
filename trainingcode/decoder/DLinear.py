import torch
import torch.nn as nn
import torch
import torch.nn as nn

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x, torch.ones_like(x)

class MultiDecoder(nn.Module):
    def __init__(self, decoder_args):
        super(MultiDecoder, self).__init__()
        self.input_dim = decoder_args.input_dim
        self.max_len = decoder_args.max_len
        self.n_classes = decoder_args.n_classes
        self.concept_dim = decoder_args.concept_dim

        decoder_args.n_classes = 1
        self.decoders = nn.ModuleList([
            SingleClassDecoder(decoder_args) for _ in range(self.n_classes)
        ])

    def forward(self, x, interp=False):
        """ 
        Args:
            x: Input tensor of shape [bs, t, concept_dim, n_classes]
        Returns:
            output: Tensor of shape [bs, pred_step, n_classes]
            attention_maps: List of attention maps for each class.
        """
        bs, t, concept_dim, n_classes = x.size()

        # Permute input to [bs, n_classes, t, concept_dim]
        x = x.permute(0, 3, 1, 2)

        outputs = []
        attention_maps = []

        for i in range(self.n_classes):
            class_input = x[:, i, :, :]
            # [bs, t, concept_dim]
            class_output, attn_map = self.decoders[i](class_input, True)  # [bs, pred_step, 1], attention_map
            outputs.append(class_output.unsqueeze(-1))
            attention_maps.append(attn_map)

        output = torch.cat(outputs, dim=-1)  # [bs, pred_step, n_classes]
        if interp:
            return output, attention_maps
        else:
            return output

class SingleClassDecoder(nn.Module):
    def __init__(self, decoder_args):
        super(SingleClassDecoder, self).__init__()
        self.input_size = decoder_args.concept_dim
        self.output_size = decoder_args.n_classes
        self.hidden_size = decoder_args.hidden_dim
        self.pred_step = decoder_args.pred_step
        
        self.linear1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size * self.pred_step, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.output_size * self.pred_step))

        self.dropout = nn.Dropout(0.03)

        self._init()
     
    def forward(self, x, interp=False):
        batch_size = x.size(0)
        
        trend = self.linear1(x)
        trend = self.linear2(trend)
        trend = self.dropout(trend)
        
        trend = trend.reshape(batch_size, -1, self.pred_step * self.output_size)
        trend_avg = trend.mean(dim=1)
        
        output = trend_avg + self.bias.view(-1, self.pred_step * self.output_size)
    
        if interp:
            return output, trend
        else:
            return output

    def _init(self, initial_bias_value=0.0):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.bias.data.fill_(initial_bias_value)
