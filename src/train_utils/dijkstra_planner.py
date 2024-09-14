"""Helper functions for training
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import random
import re
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """

    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted




class PlannerModule(pl.LightningModule):
    def __init__(self, planner, config):
        super().__init__()
        self.planner = planner
        self.config = config

    def forward(self, map_designs, goal_maps):
        inputs = torch.cat((map_designs, goal_maps), dim=1)
        return self.planner(inputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        #return torch.optim.RMSprop(self.planner.parameters(), self.config.params.lr)
        optimizer = torch.optim.Adam(self.planner.parameters(), self.config.params.lr, weight_decay=1e-4)
        steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        #return torch.optim.Adam(self.planner.parameters(), self.config.params.lr, weight_decay=1e-4)
        return [optimizer], [steplr]

    def training_step(self, train_batch, batch_idx):
        map_designs, goal_maps, opt_dist = train_batch
        outputs = self.forward(map_designs, goal_maps)
        opt_dist[abs(opt_dist + 1024) < 1] = 0
        outputs = outputs[opt_dist != 0]
        opt_dist = opt_dist[opt_dist != 0]
        loss = nn.L1Loss()(opt_dist, outputs)
        self.log("metrics/train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, goal_maps, opt_dist = val_batch
        outputs = self.forward(map_designs, goal_maps)
        opt_dist[abs(opt_dist + 1024) < 1] = 0
        outputs = outputs[opt_dist != 0]
        opt_dist = opt_dist[opt_dist != 0]
        loss = nn.L1Loss()(opt_dist, outputs)

        self.log("metrics/val_loss", loss)

        return loss


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds

    Args:
        seed (int): random seed
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)