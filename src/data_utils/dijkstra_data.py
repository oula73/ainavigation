import numpy as np
import torch.utils.data as data
from glob import glob

def create_dijkstra_dataloader(
        filename: str,
        split: str,
        batch_size: int,
        shuffle: bool = False,
) -> data.DataLoader:
    
    dataset = DijkstraDataset(filename, split)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


class DijkstraDataset(data.Dataset):
    def __init__(
            self,
            filename: str,
            split: str,
    ):
        self.filename = filename
        self.dataset_type = split

        self.map_designs, self.goal_maps, self.opt_dists = self._process(filename)

    def _process(self, filename: str):
        file_paths = glob(filename)
        map_designs_list = []
        goal_maps_list = []
        opt_dists_list = []
        dataset2idx = {"train": 0, "valid": 4, "test": 8}
        idx = dataset2idx[self.dataset_type]
        for path in file_paths:
            with np.load(path) as f:
                map_designs_list.append(f["arr_" + str(idx)])
                goal_maps_list.append(f["arr_" + str(idx + 1)])
                opt_dists_list.append(f["arr_" + str(idx + 3)])

        map_designs = np.concatenate(map_designs_list, axis=0).astype(np.float32)
        goal_maps = np.concatenate(goal_maps_list, axis=0).astype(np.float32)
        opt_dists = np.concatenate(opt_dists_list, axis=0).astype(np.float32)
        
        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(map_designs.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(map_designs.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(map_designs.shape[0]))
        print("\tSize: {}x{}".format(map_designs.shape[1], map_designs.shape[2]))
        return map_designs, goal_maps, opt_dists
    

    def __getitem__(self, index: int):
        map_design = self.map_designs[index][np.newaxis]
        goal_map = self.goal_maps[index]
        opt_dist = self.opt_dists[index]

        return map_design, goal_map, opt_dist

    def __len__(self):
        return self.map_designs.shape[0]