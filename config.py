'''
Constants and parameters :)
'''

import torch
from torch.optim import Adam
import torchvision.transforms as transforms

img_size = 256
batch_size = 32
lr = 3e-4
n_iters = 1000
rootdir = '.'
datapath = './alice_dataset'
checkpoint_path = './checkpoints'
checkpoint_every = None
optimizer = Adam
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.Resize(300),
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])