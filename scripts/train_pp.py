from __future__ import annotations

import os
import sys
import hydra
import torch
import pytorch_lightning as pl
import torch.nn as nn

sys.path.append(r"./src")
from module import CNND, UnetD
from data_utils.PP_data import create_PP_dataloader
from train_utils.pp_planner import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint
from train_utils.pp_planner import load_from_ptl_checkpoint
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms


@hydra.main(config_path="config", config_name="train_pp")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_PP_dataloader(
        config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_PP_dataloader(
        config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
    )

    # encoder = vit_b_16()
    # encoder.conv_proj = nn.Conv2d(2, 768, kernel_size=(16, 16), stride=(16, 16))
    # encoder.heads = nn.Sequential(
    #     nn.Linear(encoder.heads[0].in_features, 32 * 32),
    #     nn.Sigmoid(),
    # )
    
    encoder = UnetD(input_dim=2, encoder_depth=4)

    #neural_astar.load_state_dict(load_from_ptl_checkpoint("model/all_data/lightning_logs"))

    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/val_loss", save_weights_only=True, mode="min"
    )

    module = PlannerModule(encoder, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset.replace("*", "pp_data"))}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()