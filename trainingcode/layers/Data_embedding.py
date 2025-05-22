from torch import nn
import torch
import math


class SelectiveTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, selected_channels=None):  # selected_channels: [1, 0, 1, 1, ..., 0]
        super(SelectiveTokenEmbedding, self).__init__()
        if selected_channels is None:
            selected_channels = list(range(c_in))
        self.d_model = d_model
        selected_channels = {i: selected_channels[i] for i in range(len(selected_channels))}
        self.selected_channels = [i for i in selected_channels if selected_channels[i] == 1]
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=len(self.selected_channels), out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU 

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.to(self.tokenConv.weight.device)  # Ensure input is on the same device as weights
        # Select channels and permute dimensions
        x = x[:, :, :, self.selected_channels]  # Select channels, shape: [bs, t1, t2, selected_channels]
        bs, t1, t2, c_in = x.size()
        x = x.permute(0, 3, 1, 2)  # Change shape to [bs, selected_channels, t1, t2]
        x = x.reshape(x.size(0), x.size(1), -1)  # Reshape to [bs, selected_channels, t1*t2]
        
        # Apply Conv1d
        x = self.tokenConv(x)  # Shape: [bs, d_model, t1*t2]
        
        # Reshape back to [bs, t1, t2, d_model]
        x = x.reshape(bs,self.d_model, t1, t2)
        x = x.permute(0, 2, 3, 1)
        return x

class AllTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(AllTokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU 
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.to(self.tokenConv.weight.device)  # Ensure input is on the same device as weights
        if x.dim() == 4:
            x_after = []
            for i in range(x.size(1)):
                x_after.append(self.tokenConv(x[:, i, :, :].permute(0, 2, 1)).transpose(1, 2))
            x = torch.stack(x_after, dim=1)
        else:
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 计算位置编码一次（对数空间）
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(-2)
        pe_slice = self.pe[:, :seq_len]
        if x.dim() == 4:
            return pe_slice.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        else:
            return pe_slice

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, selected_channels=None):
        super(DataEmbedding, self).__init__()
        if selected_channels is None:
            self.value_embedding = AllTokenEmbedding(c_in=c_in, d_model=d_model)
        else:
            self.value_embedding = SelectiveTokenEmbedding(c_in=c_in, d_model=d_model, selected_channels=selected_channels)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        # move to GPU
        self.value_embedding.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.position_embedding.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout.to('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        x = x.to(self.value_embedding.tokenConv.weight.device)  # Ensure input is on the same device as weights
        x = self.value_embedding(x) + self.position_embedding(x)
        return x
    # self.dropout(x)
    
def test_data_embedding():
    batch_size = 4
    seq_len = 10
    c_in = 5
    d_model = 32
    selected_channels = [0, 2, 4]

    x = torch.randn(batch_size, seq_len, c_in)

    data_embedding = DataEmbedding(c_in=c_in, d_model=d_model, selected_channels=selected_channels)
    output = data_embedding(x)
    print(output.shape)

if __name__ == "__main__":
    test_data_embedding()