import torch
import torch.nn as nn

class TimeEncoder(nn.Module):
    def __init__(self, d_pe: int, max_len: int = 512):
        """
        通过位置编码隐式捕获时间信息
        
        参数:
            d_pe (int): 位置编码维度
            max_len (int): 支持的最大序列长度
        """
        super().__init__()
        self.d_pe = d_pe
        
        # 生成固定位置编码（不可学习）
        pe = torch.zeros(max_len, d_pe)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_pe, 2).float() * (-math.log(10000.0) / d_pe)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos
        pe = pe.unsqueeze(0)  # 增加batch维度 -> [1, max_len, d_pe]
        self.register_buffer("pe", pe)  # 注册为不可训练参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: [batch_size, seq_len, feature_dim]
        输出:
            [batch_size, seq_len, feature_dim] 带位置编码的特征
        """
        x = x + self.pe[:, :x.size(1), :]  # 自动广播到batch维度
        return x

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=None):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size

    def moving_avg(self, x, kernel_size):
        padding = (kernel_size - 1) // 2
        return nn.functional.avg_pool1d(x, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        """
        Decompose the input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels]
            
        Returns:
            tuple: (residual, trend)
                - residual: The residual component of shape [batch_size, seq_len, channels]
                - trend: The trend component of shape [batch_size, seq_len, channels]
        """
        batch_size, seq_len, channels = x.size()
        
        # Automatically select kernel_size if not provided
        if self.kernel_size is None:
            self.kernel_size = max(3, seq_len // 10)  # Example heuristic: 1/10th of the sequence length, minimum 3
        
        # Ensure kernel_size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        # Reshape for 1D convolution
        x_reshaped = x.reshape(batch_size * channels, 1, seq_len)
        
        # Calculate moving average (trend)
        trend = self.moving_avg(x_reshaped, self.kernel_size)
        
        # Reshape back
        trend = trend.reshape(batch_size, channels, seq_len).transpose(1, 2)
        
        # Calculate residual
        residual = x - trend
        
        return residual, trend

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    batch_size, seq_len, channels = 32, 100, 3
    x = torch.randn(batch_size, seq_len, channels)
    
    # 初始化分解模块
    decomp = SeriesDecomposition()  # 不指定 kernel_size
    
    # 进行分解
    residual, trend = decomp(x)
    
    # 验证输出形状
    print(f"Input shape: {x.shape}")
    print(f"Residual shape: {residual.shape}")
    print(f"Trend shape: {trend.shape}")