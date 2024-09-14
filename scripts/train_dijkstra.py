from __future__ import annotations

import os
import sys
import hydra
import torch
import pytorch_lightning as pl

sys.path.append(r"./src")
from module import CNND, UnetD
from data_utils.dijkstra_data import create_dijkstra_dataloader
from train_utils.dijkstra_planner import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint
from train_utils.dijkstra_planner import load_from_ptl_checkpoint

@hydra.main(config_path="config", config_name="train_dijkstra")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dijkstra_dataloader(
        config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_dijkstra_dataloader(
        config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
    )

    encoder = UnetD(input_dim=2, encoder_depth=4)

    #neural_astar.load_state_dict(load_from_ptl_checkpoint("model/all_data/lightning_logs"))

    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/val_loss", save_weights_only=True, mode="min"
    )

    module = PlannerModule(encoder, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset.replace("*", "all_data"))}"
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