import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=3, kernel_size=3, dropout=0.0):
        super(TCNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        nn.init.xavier_uniform_(self.input_proj[0].weight) 
        nn.init.zeros_(self.input_proj[0].bias)            
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            self.blocks.append(
                TCNBlock(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        # 输出模块
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        bs, t, n = x.shape

        x = self.input_proj(x)          # [bs, t, h]
        x = x.permute(0, 2, 1)          # [bs, h, t]
        
        for block in self.blocks:
            x = block(x, t)          
            
        x = self.global_pool(F.leaky_relu(x))  # [bs, h, 1]
        x = x.squeeze(-1)                      # [bs, h]
        x = self.output_proj(x)                # [bs, o]
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return x

class TCNBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, padding, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size,
            padding=padding, dilation=dilation,
            groups=hidden_size  
        )
        self.norm = nn.LayerNorm(hidden_size) 
        self.act = nn.LeakyReLU() 
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1)) 

    def forward(self, x, seq_len):
        residual = x
        
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [bs, t, h]
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1) 
        
        x = x[..., :seq_len]
        return residual + self.gamma * x 