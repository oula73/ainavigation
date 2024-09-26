import torch
import torch.nn as nn
import numpy as np
from dataloader.mapsample import MapSample
from model.pe import GaussianRelativePE

def get_gaussian(k=3, sigma=0, normalized=True):
    if sigma == 0:
        sigma = k / 5
    sigma_square = sigma ** 2
    coord_x = np.stack([np.arange(-(k // 2), k // 2 + 1) for _ in range(k)])
    coord_y = coord_x.T
    alpha = 1 / (2 * np.pi * sigma_square)
    out = alpha * np.exp(- 1 / (2 * sigma_square) * (coord_x ** 2  + coord_y ** 2))
    if normalized:
        out /= out.sum()
    return out

class CCCNet(nn.Module):
    def __init__(self, n_layers=3, gaussian_blur_kernel=0):
        super().__init__()

        class convBlock(nn.Module):
            def __init__(self, in_channels, out_channels, activation=None):
                super().__init__()
                self.activation = activation if activation is not None else nn.ReLU()
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.bn3 = nn.BatchNorm2d(out_channels * 2)
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
                self.conv3 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=2)
                self.fw = nn.Sequential(
                    self.conv1, self.bn1, self.activation,
                    self.conv2, self.bn2, self.activation,
                    self.conv3, self.bn3, self.activation,
                )

            def forward(self, x):
                return self.fw(x)
    
        self.gaussian_blur_kernel= gaussian_blur_kernel
        if gaussian_blur_kernel > 0:
            gaussian_kernel = torch.tensor(get_gaussian(gaussian_blur_kernel, sigma=0, normalized=True), dtype=torch.float32).view(1, 1, gaussian_blur_kernel, gaussian_blur_kernel)
            self.blur = nn.Conv2d(1, 1, gaussian_blur_kernel, padding=gaussian_blur_kernel // 2, bias=False)
            self.blur.weight.data = gaussian_kernel
        else:
            self.blur = None

        self.pe = GaussianRelativePE(100)
        self.sigm = nn.Sigmoid()
        n_channels = [64 * (2 ** i) for i in range(n_layers)]
        self.conv_down = nn.ModuleList([convBlock(c if i > 0 else 3, c) for i, c in enumerate(n_channels)])
        self.bottleneck = nn.Conv2d(512, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.Linear = nn.Linear(in_features=4096, out_features=20)

    def pe_forward(self, x, start, goal):
        zeros = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        pe_start = self.pe(zeros, start)
        pe_goal = self.pe(zeros, goal)
        return torch.cat([x, pe_start, pe_goal], dim=1)
    
    def forward(self, x, start, goal):
        if self.gaussian_blur_kernel > 0:
            with torch.no_grad():
                x = self.blur(x)
        x = self.pe_forward(x, start, goal)
        for i, conv in enumerate(self.conv_down):
            x = conv(x)
        x = self.bottleneck(x)
        x = self.Linear(torch.flatten(x, start_dim=1))
        x = self.sigm(x)
        return x