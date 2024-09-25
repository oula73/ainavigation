import torch
import os
from configuration import config
from dataloader.dataset import MapDataset
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    maps = torch.stack([sample[0].map for sample in batch]).unsqueeze(1)
    starts = torch.stack([sample[0].start for sample in batch])
    goals = torch.stack([sample[0].goal for sample in batch])
    paths = []
    for s in batch:
        sample = s[0]
        path = torch.zeros_like(sample.map, dtype=sample.map.dtype)
        path[sample.path[:, 0], sample.path[:, 1]] = 1.0
        paths.append(path)
    paths = torch.stack(paths)
    return maps, starts, goals, paths

def custom_collate_fn_extended(batch):
    filenames = [sample[1] for sample in batch]
    path_array = [sample[0].path for sample in batch]
    out = list(custom_collate_fn(batch))
    out.append(filenames)
    out.append(path_array)
    return out

def custom_collate_fn_wpt(batch):
    maps = torch.stack([sample[0].map for sample in batch]).unsqueeze(1)
    goals = torch.stack([sample[0].goal for sample in batch])
    path_arrays = [sample[0].path for sample in batch]
    path_arrays = torch.nn.utils.rnn.pad_sequence(path_arrays, batch_first=True, padding_value=0.0)
    return maps, goals, path_arrays

def get_dataLoader(split, collate_fn=custom_collate_fn):
    if split == "train":
        dataset = MapDataset(datapath=os.path.join(config.raw_data_path, "train"), lazy=False)
        loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    elif split == "valid":
        dataset = MapDataset(datapath=os.path.join(config.raw_data_path, "valid"), lazy=False)
        loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    elif split == "test":
        dataset = MapDataset(datapath=os.path.join(config.raw_data_path, "test"), lazy=True)
        loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    elif split == "train_debug":
        dataset = MapDataset(datapath=os.path.join(config.raw_data_path, "debug"), lazy=True)
        loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    elif split == "test_debug":
        dataset = MapDataset(datapath=os.path.join(config.raw_data_path, "debug"), lazy=True)
        loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn_extended)
    else:
        raise ValueError("split name must be train, valid or test.")
    return loader