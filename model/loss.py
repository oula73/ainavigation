# import torch
from torch import nn
from torch.nn import MSELoss
import torch


class MSEMapLoss(nn.Module):
    def __init__(self):
        super(MSEMapLoss, self).__init__()
        self._loss = MSELoss()

    def forward(self, x, y):
        return self._loss(x, y)
    
class MSEPathLoss(nn.Module):
    def __init__(self):
        super(MSEPathLoss, self).__init__()
        self._loss = MSELoss(reduce='none')

    def forward(self, predict_path, truth_path):
        truth_path_shifted = torch.cat([truth_path[:, 1:, :], torch.zeros_like(truth_path[:, :1, :])], dim=1)
        mask = (torch.sum(truth_path_shifted, dim=-1) != 0)
        loss = self._loss(predict_path, truth_path_shifted / 100.0)
        loss = torch.mean(loss[mask])
        return loss