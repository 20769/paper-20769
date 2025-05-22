import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from argparse import Namespace

class MultiDecoder(nn.Module):
    def __init__(self, decoder_args):
        super(MultiDecoder, self).__init__()
        self.input_dim = decoder_args.input_dim
        self.max_len = decoder_args.max_len
        self.n_classes = decoder_args.n_classes
        self.concept_dim = decoder_args.concept_dim

        single_decoder_args = Namespace(
            concept_dim=self.concept_dim,
            hidden_dim=decoder_args.hidden_dim,
            n_classes=1,
            pred_step=decoder_args.pred_step
        )
        
        # Create a separate decoder for each class
        self.decoders = nn.ModuleList([
            SingleClassDecoder(single_decoder_args) for _ in range(self.n_classes)
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

        # Decode independently for each class
        for i in range(self.n_classes):
            class_input = x[:, i, :, :]  # [bs, t, concept_dim]
            class_output, attn_map = self.decoders[i](class_input, True)  # [bs, pred_step, 1], attention_map
            outputs.append(class_output)
            attention_maps.append(attn_map)

        # Concatenate outputs for all classes
        output = torch.cat(outputs, dim=-1)  # [bs, pred_step, n_classes]
        if interp:
            return output, attention_maps
        else:
            return output

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.1):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                    dilation=dilation_size, padding=(kernel_size-1)*dilation_size//2,
                                    dropout=dropout)]
        
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.transpose(1, 2)  # -> (batch_size, input_size, seq_len)
        y = self.tcn(x)
        y = self.linear(y.transpose(1, 2))  # -> (batch_size, seq_len, output_size)
        return y
    
class ResidualGate(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualGate, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear1(x)) * x + self.linear2(x)

class SingleClassDecoder(nn.Module):
    def __init__(self, decoder_args):
        super(SingleClassDecoder, self).__init__()
        self.pred_step = decoder_args.pred_step
        self.tcn = TCN(
            input_size=decoder_args.concept_dim,
            output_size=decoder_args.concept_dim,
            num_channels=[decoder_args.hidden_dim] * 3,  # 3 layers TCN
            kernel_size=3,
            dropout=0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(decoder_args.concept_dim, decoder_args.concept_dim),  # 维度对齐
            nn.GELU(),
            nn.Linear(decoder_args.concept_dim, decoder_args.concept_dim)  # 输出维度与输入x一致
        )

        self.fc = nn.Sequential(
            nn.Linear(decoder_args.hidden_dim, decoder_args.hidden_dim),
            nn.LayerNorm(decoder_args.hidden_dim),  # Add normalization
            nn.GELU(),  # Replace ReLU with GELU
            nn.utils.parametrizations.spectral_norm(  # Spectral norm on output layer
                nn.Linear(decoder_args.hidden_dim, decoder_args.n_classes),
                n_power_iterations=1,
                eps=1e-4 
            )
        )
        self.linear = nn.Linear(decoder_args.concept_dim, decoder_args.hidden_dim)
        self.dropout = nn.Dropout(0.0)
        self.res_gate = ResidualGate(decoder_args.hidden_dim)
        self.layer_norm = nn.LayerNorm(decoder_args.concept_dim)
        self.pos_encoder = nn.Embedding(1000, decoder_args.hidden_dim)
        self.temporal_proj = nn.Sequential(
            nn.Linear(decoder_args.hidden_dim, 4*decoder_args.hidden_dim),
            nn.GELU(),
            nn.Linear(4*decoder_args.hidden_dim, decoder_args.hidden_dim)
        )
        self.projection = nn.Linear(decoder_args.concept_dim, decoder_args.concept_dim)
        self._init()

    def forward(self, x, interp=False):
        # Use TCN instead of LSTM
        tcn_out = self.tcn(x)  # [batch_size, seq_len, hidden_dim]
        projected = self.projection(tcn_out)  # [bs, seq_len, concept_dim]
        attn_out = self.attention(projected)  # [bs, seq_len, concept_dim]
        fused = self.layer_norm(attn_out + tcn_out)
        fused = self.linear(fused)

        if self.pred_step <= fused.size(1):
            windows = fused.unfold(1, self.pred_step, 1)
            last_outputs = windows.mean(dim=1)
        else:
            final_state = fused[:, -1]
            expanded = final_state.unsqueeze(1).repeat(1, self.pred_step, 1)
            positions = self.pos_encoder(torch.arange(self.pred_step, device=x.device)).detach()
            expanded += positions.unsqueeze(0)
            last_outputs = self.temporal_proj(expanded)
        if torch.isnan(last_outputs).any():
            print("Nan detected in last_outputs")
        if self.pred_step == 1:
            last_outputs = last_outputs.squeeze(-1)
        if interp:
            return self.fc(last_outputs), attn_out
        else:
            return self.fc(last_outputs)
    
    def _init(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
