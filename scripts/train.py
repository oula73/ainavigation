from __future__ import annotations

import os
import sys
import hydra
import torch
import pytorch_lightning as pl

sys.path.append(r"./src")
from module import NeuralAstar
from data_utils.maze_data import create_dataloader
from train_utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path="config", config_name="train")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
    )

    neural_astar = NeuralAstar(
        encoder_input=config.encoder.input,
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        learn_obstacles=False,
        Tmax=config.Tmax,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )

    module = PlannerModule(neural_astar, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
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
