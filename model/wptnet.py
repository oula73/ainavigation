import math
import torch
from torch import nn
import torch.nn.functional as F
from dataloader.mapsample import MapSample
import numpy as np
from model.pe import GaussianRelativePE

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, norm_first=False, transpose=False, last_output_padding=0):
        super(ConvBlock, self).__init__()
        self.activation = activation if activation is not None else nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if transpose:               
            self.bn3 = nn.BatchNorm2d(out_channels // 2)     
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3)
            self.conv3 = nn.ConvTranspose2d(out_channels, out_channels // 2, 3, stride=2, padding=2, output_padding=last_output_padding)
        else:
            self.bn3 = nn.BatchNorm2d(out_channels * 2)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
            self.conv3 = nn.Conv2d(out_channels, out_channels * 2, 3, stride=2)
        if norm_first:
            self._fw = nn.Sequential(
                self.conv1, self.bn1, self.activation, 
                self.conv2, self.bn2, self.activation,
                self.conv3, self.bn3, self.activation)
        else:
            self._fw = nn.Sequential(
                self.conv1, self.activation, self.bn1,
                self.conv2, self.activation, self.bn2,
                self.conv3, self.activation, self.bn3)

    def forward(self, x):
        return self._fw(x)
    
class EmbeddingNet(nn.Module):
    def __init__(self, n_layers=3):
        super(EmbeddingNet, self).__init__()
        n_channels = [64 * (2 ** i) for i in range(n_layers)]
        self.conv_blocks = nn.ModuleList([ConvBlock(c if i > 0 else 3, c) for i, c in enumerate(n_channels)])
    
    def forward(self, x):
        multscale_feature = []
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            multscale_feature.append(F.adaptive_avg_pool2d(x, 1))
        summary_feature = torch.cat(multscale_feature, dim=1)
        return summary_feature

def positional_encoding(sequence_length, embedding_dim):
    # 创建一个维度为(sequence_length, embedding_dim)的位置嵌入矩阵
    position = torch.arange(0, sequence_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros(sequence_length, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    pos_encoding = pos_encoding.unsqueeze(0)  # 增加一个批次维度
    return pos_encoding

class WPTNet(nn.Module):
    def __init__(self, n_layers=3):
        super(WPTNet, self).__init__()
            
        self.pe = GaussianRelativePE(100)
        self.conv = EmbeddingNet()
        encoder_layer = nn.TransformerEncoderLayer(d_model=64 * (2 + 4 + 8), nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.head = nn.Sequential(
            nn.Linear(64 * (2 + 4 + 8), 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def pe_forward(self, x, current, goal):
        batch_size, length, _ = current.shape
        _, channel, height, width = x.shape
        current = current.view(batch_size * length, -1)
        zeros = torch.zeros((batch_size*length, 1, height, width), device=x.device, dtype=x.dtype)

        pe_current = self.pe(zeros, current)
        pe_goal = self.pe(zeros, goal.repeat(length, 1))

        x = x.unsqueeze(dim=1)
        x = x.repeat(1, length, 1, 1, 1)
        pe_current = pe_current.view(batch_size, length, channel, height, width)
        pe_goal = pe_goal.view(batch_size, length, channel, height, width)

        return torch.cat([x, pe_current, pe_goal], dim=2)

    def forward(self, x, current, goal):
        """Forward pass

        Args:
            x (Tensor): (N, C, H, W) batch of 2d maps.
            current (Tensor): (N, L, 2) current position.
            goal (Tensor)): (N, 2) goal position.

        Returns:
            Tensor: path way points.
        """
        batch_size, _, height, width = x.shape
        _, length, _  = current.shape

        causal_mask = torch.triu(torch.ones(length, length), diagonal=1).bool().to(x.device)
        src_key_padding_mask = (torch.sum(current, dim=2) == 0).detach()

        x = self.pe_forward(x, current, goal) # (N, L, C + 2, H, W)

        x = x.view(batch_size * length, -1, height, width)
        x = self.conv(x)
        x = x.view(batch_size, length, -1)

        position_embedding = positional_encoding(length, x.shape[2])
        x = x + position_embedding

        hidden = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask, is_causal=True)
        out = self.head(hidden)
        return out

if __name__ == "__main__":
    model = WPTNet()
    path = r'E:\code\navigation\raw_data\valid\0a2ae4a1-72e8-4daf-9b29-dcd6a24b5af6.pt'
    sample = MapSample.load(path)
    print(sample.map.shape, sample.start.shape, sample.goal.shape, sample.path.shape)
    pass