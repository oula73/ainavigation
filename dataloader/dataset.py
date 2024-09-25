import os
import sys, types
from tqdm import tqdm
from torch.utils.data import Dataset
from random import shuffle, random
from dataloader.mapsample import MapSample

def collect_files(datapath):
    for entry in os.scandir(os.path.abspath(datapath)):
        if os.path.isfile(entry.path):
            yield entry.path
        elif os.path.isdir(entry.path):
            for nested_entry in collect_files(entry.path):
                yield nested_entry

class MapDataset(Dataset):
    def __init__(self, datapath, lazy=True):
        super(MapDataset, self).__init__()
        datapath = os.path.abspath(datapath)
        self.samples = list(collect_files(datapath))
        shuffle(self.samples)
        self._lazy = lazy

        sys.modules['dataset'] = types.ModuleType('dataset')
        sys.modules['dataset.map_sample'] = types.ModuleType('map_sample')
        sys.modules['dataset.map_sample'].__dict__['MapSample'] = MapSample

        if not lazy:
            self.samples = [MapSample.load(sample) for sample in tqdm(self.samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._lazy:
            return MapSample.load(self.samples[idx]), self.samples[idx]
        else:
            return self.samples[idx], self.samples[idx]