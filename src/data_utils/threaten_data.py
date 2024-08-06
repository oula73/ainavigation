import numpy as np
import torch.utils.data as data
from glob import glob

def create_threat_dataloader(
        filename: str,
        split: str,
        batch_size: int,
        shuffle: bool = False,
) -> data.DataLoader:
    
    dataset = ThreatDataset(filename, split)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


class ThreatDataset(data.Dataset):
    def __init__(
            self,
            filename: str,
            split: str,
            pct1: float = 0.55,
            pct2: float = 0.70,
            pct3: float = 0.85,
    ):
        self.filename = filename
        self.dataset_type = split

        self.map_designs, self.goal_maps, self.threat_maps, self.opt_policies, self.opt_dists = self._process(filename)
        self.pcts = np.array([pct1, pct2, pct3, 1.0])

    def _process(self, filename: str):
        file_paths = glob(filename)
        map_designs_list = []
        goal_maps_list = []
        threat_maps_list = []
        opt_policies_list = []
        opt_dists_list = []
        dataset2idx = {"train": 0, "valid": 5, "test": 10}
        idx = dataset2idx[self.dataset_type]
        for path in file_paths:
            with np.load(path) as f:
                map_designs_list.append(f["arr_" + str(idx)])
                goal_maps_list.append(f["arr_" + str(idx + 1)])
                threat_maps_list.append(f["arr_" + str(idx + 2)])
                opt_policies_list.append(f["arr_" + str(idx + 3)])
                opt_dists_list.append(f["arr_" + str(idx + 4)])

        map_designs = np.concatenate(map_designs_list, axis=0).astype(np.float32)
        goal_maps = np.concatenate(goal_maps_list, axis=0).astype(np.float32)
        threat_maps = np.concatenate(threat_maps_list, axis=0).astype(np.float32)
        opt_policies = np.concatenate(opt_policies_list, axis=0).astype(np.float32)
        opt_dists = np.concatenate(opt_dists_list, axis=0).astype(np.float32)
        
        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(map_designs.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(map_designs.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(map_designs.shape[0]))
        print("\tSize: {}x{}".format(map_designs.shape[1], map_designs.shape[2]))
        return map_designs, goal_maps, threat_maps, opt_policies, opt_dists
    

    def __getitem__(self, index: int):
        map_design = self.map_designs[index][np.newaxis]
        goal_map = self.goal_maps[index]
        threat_map = self.threat_maps[index]
        opt_policy = self.opt_policies[index]
        opt_dist = self.opt_dists[index]
        start_map = self.get_random_start_map(opt_dist)
        opt_traj = self.get_opt_traj(start_map, goal_map, opt_policy)

        return map_design, start_map, goal_map, threat_map, opt_traj

    def __len__(self):
        return self.map_designs.shape[0]


    def get_opt_traj(
        self, start_map: np.array, goal_map: np.array, opt_policy: np.array
    ) -> np.array:
        """
        Get optimal path from start to goal using pre-computed optimal policy

        Args:
            start_map (np.array): one-hot matrix for start location
            goal_map (np.array): one-hot matrix for goal location
            opt_policy (np.array): optimal policy to arrive at goal location from arbitrary locations

        Returns:
            np.array: optimal path
        """

        opt_traj = np.zeros_like(start_map)
        opt_policy = opt_policy.transpose((1, 2, 3, 0))
        current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
        while goal_loc != current_loc:
            opt_traj[current_loc] = 1.0
            next_loc = self.next_loc(current_loc, opt_policy[current_loc])
            assert (
                opt_traj[next_loc] == 0.0
            ), "Revisiting the same position while following the optimal policy"
            current_loc = next_loc

        return opt_traj
    
    def get_random_start_map(self, opt_dist: np.array) -> np.array:
        """
        Get random start map
        This function first chooses one of 55-70, 70-85, and 85-100 percentile intervals.
        Then it picks out a random single point from the region in the selected interval.

        Args:
            opt_dist (np.array): optimal distances from each location to goal

        Returns:
            np.array: one-hot matrix of start location
        """
        od_vct = opt_dist.flatten()
        od_vals = od_vct[od_vct > od_vct.min()]
        od_th = np.percentile(od_vals, 100.0 * (1 - self.pcts))
        r = np.random.randint(0, len(od_th) - 1)
        start_candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
        start_idx = np.random.choice(np.where(start_candidate)[0])
        start_map = np.zeros_like(opt_dist)
        start_map.ravel()[start_idx] = 1.0

        return start_map

    def next_loc(self, current_loc: tuple, one_hot_action: np.array) -> tuple:
        """
        Choose next location based on the selected action

        Args:
            current_loc (tuple): current location
            one_hot_action (np.array): one-hot action selected by optimal policy

        Returns:
            tuple: next location
        """
        action_to_move = [
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
            (0, +1, 0),
            (0, -1, +1),
            (0, -1, -1),
            (0, +1, +1),
            (0, +1, -1),
        ]
        move = action_to_move[np.argmax(one_hot_action)]
        return tuple(np.add(current_loc, move))