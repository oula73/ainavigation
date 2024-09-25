import torch
from dataloader.mapsample import MapSample
from model.wptnet import WPTNet
from matplotlib import pyplot as plt
import types
import sys

sys.modules['dataset'] = types.ModuleType('dataset')
sys.modules['dataset.map_sample'] = types.ModuleType('map_sample')
sys.modules['dataset.map_sample'].__dict__['MapSample'] = MapSample

model = WPTNet()
path = r'E:\code\navigation\raw_data\valid\0a2ae4a1-72e8-4daf-9b29-dcd6a24b5af6.pt'
sample = MapSample.load(path)
print(sample.map.shape, sample.start.shape, sample.goal.shape, sample.path.shape)
a = model(sample.map.unsqueeze(dim=0).unsqueeze(dim=0), sample.path.unsqueeze(dim=0), sample.goal.unsqueeze(dim=0))
print(a.shape)

for i in range(a.shape[1]):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(a[0, i, 0, :, :])
    plt.subplot(1,3,2)
    plt.imshow(a[0, i, 1, :, :])
    plt.subplot(1,3,3)
    plt.imshow(a[0, i, 2, :, :])
    plt.show()