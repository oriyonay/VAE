'''
misc utilities
'''

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid

def infinite_dataloader(dataloader):
    '''
    a neat function for getting the next batch without epochs
    '''
    while True:
        for data in dataloader:
            yield data

def get_data(datapath, transform, batch_size):
    dataset = datasets.ImageFolder(datapath, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = infinite_dataloader(dataloader)
    return dataloader

def plot_images(images):
    plt.figure(figsize=(32, 32))
    images = torch.cat([i for i in images.cpu()], dim=-1)
    images = torch.cat([images], dim=-2).permute(1, 2, 0).cpu()
    plt.imshow(images)
    plt.show()

def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    arr = grid.permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(arr)
    im.save(path)