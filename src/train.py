import os
import glob
import random
import argparse
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

from utils import seed_everything, ImageTransform, ImageTransform_2, PANDADataset, Trainer
from model import ModelEFN


# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname')
parser.add_argument('-model', '--model_name', default='b0')
parser.add_argument('-trn_s', '--train_size', type=float, default=0.8)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr', '--lr', type=float, default=0.0005)
parser.add_argument('-ims', '--image_size', type=int, default=224)
parser.add_argument('-t_ims', '--tile_image_size', type=int, default=224)
parser.add_argument('-img_n', '--img_num', type=int, default=12)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)
parser.add_argument('--tile', action='store_true')
parser.add_argument('-tiff_l', '--tiff_level', type=int, default=-1)
parser.add_argument('-sch', '--scheduler', choices=['step', 'cos'], default='step')

arges = parser.parse_args()
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config  ################################################################
config = {
    'tiff_level': arges.tiff_level,
    'img_size': arges.image_size,
    'use_tile': arges.tile,
    'img_num': arges.img_num,
    'tile_img_size': arges.tile_image_size
}

train_size = arges.train_size
batch_size = arges.batch_size
lr = arges.lr
num_epochs = arges.num_epoch
exp_name = arges.expname
model_name = f'efficientnet-{arges.model_name}'

# Data Loading  ################################################################
data_dir = '../data/input/train_images'
meta = pd.read_csv('../data/input/train.csv')
transform = ImageTransform_2(config['img_size'])
meta = meta.sample(frac=1.0).reset_index(drop=True)
# Train Validation Split
train = meta.iloc[:int(len(meta) * train_size)]
val = meta.iloc[int(len(meta) * train_size):]
del meta

# Dataset, DataLoader  ################################################################
train_dataset = PANDADataset(train, data_dir, 'train', transform, **config)
val_dataset = PANDADataset(val, data_dir, 'val', transform, **config)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

print('Data Num')
print('Train: ', len(dataloaders['train']))
print('Val: ', len(dataloaders['val']))
print('Use Tile: ', arges.tile)

# Model  ################################################################
net = ModelEFN(model_name=model_name, output_size=6)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction='mean')
sch_dict = {
    'step': StepLR(optimizer, step_size=5, gamma=0.5),
    'cos': CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.1)
}
scheduler = sch_dict[arges.scheduler]

# Train  ################################################################
trainer = Trainer(dataloaders, net, device, num_epochs, criterion, optimizer, scheduler,
                  batch_multiplier=5, exp=exp_name)
trainer.train()
