"""Customized dataset
Author: Ryo Yonetani, Mohammadamin Barekatain
Affiliation: OSX
"""

from __future__ import annotations, print_function

import numpy as np
import torch.utils.data as data

def create_warcraft_dataloader(
    dirname: str,
    split: str,
    batch_size: int,
    shuffle: bool = False,
) -> data.DataLoader:
    """
    Create dataloader from npz file

    Args:
        dirname (str): directory name
        split (str): data split: either train, valid, or test
        batch_size (int): batch size
        shuffle (bool, optional): whether to shuffle samples. Defaults to False.

    Returns:
        torch.utils.data.DataLoader: dataloader
    """

    dataset = WarCraftDataset(dirname, split)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


class WarCraftDataset(data.Dataset):
    def __init__(
        self,
        dirname: str,
        split: str,
    ):
        self.map_designs = (
            (np.load(f"{dirname}/{split}_maps.npy").transpose(0, 3, 1, 2) / 255.0)
        ).astype(np.float32)
        self.paths = np.load(f"{dirname}/{split}_shortest_paths.npy").astype(np.float32)

    def __getitem__(self, index: int):
        map_design = self.map_designs[index]
        opt_traj = self.paths[index][np.newaxis]
        start_map = np.zeros_like(opt_traj)
        start_map[:, 0, 0] = 1
        goal_map = np.zeros_like(opt_traj)
        goal_map[:, -1, -1] = 1

        return map_design, start_map, goal_map, opt_traj

    def __len__(self):
        return self.map_designs.shape[0]
