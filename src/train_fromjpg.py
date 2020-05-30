import os
import glob
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

from utils import seed_everything, ImageTransform, ImageTransform_2, PANDADataset_4, Trainer
from model import ModelEFN

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname')
parser.add_argument('-model', '--model_name', default='b0')
parser.add_argument('-trn_s', '--train_size', type=float, default=0.8)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr', '--lr', type=float, default=0.0005)
parser.add_argument('-ims', '--image_size', type=int, default=224)
parser.add_argument('-img_n', '--img_num', type=int, default=12)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)
parser.add_argument('-sch', '--scheduler', choices=['step', 'cos', 'none'], default='step')

arges = parser.parse_args()
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config  ################################################################
config = {
    'img_num': arges.img_num,
    'img_size': arges.image_size
}

train_size = arges.train_size
batch_size = arges.batch_size
lr = arges.lr
num_epochs = arges.num_epoch
exp_name = arges.expname
model_name = f'efficientnet-{arges.model_name}'

# Data Loading  ################################################################
img_path = glob.glob('../data/grid_224_2/*.jpg')
meta = pd.read_csv('../data/input/train.csv')
transform = ImageTransform_2(config['img_size'])

# idごとの画像数を抽出しimg_numより少ないimgは対象外にする
img_id = [s.split(sep)[-1].split('_')[0] for s in img_path]
u, count = np.unique(img_id, return_counts=True)
img_id = u[count > config['img_num']]
meta = meta[meta['image_id'].isin(img_id)]

# Train Validation Split
meta = meta.sample(frac=1.0).reset_index(drop=True)
train = meta.iloc[:int(len(meta) * train_size)]
val = meta.iloc[int(len(meta) * train_size):]
del meta

# Dataset, DataLoader  ################################################################
train_dataset = PANDADataset_4(img_path, train, 'train', transform, **config)
val_dataset = PANDADataset_4(img_path, val, 'val', transform, **config)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

print('Data Num')
print('Train: ', len(dataloaders['train'].dataset))
print('Val: ', len(dataloaders['val'].dataset))
print('#'*30)
print('Iterarion')
print('Train: ', len(dataloaders['train']))
print('Val: ', len(dataloaders['val']))

# Model  ################################################################
net = ModelEFN(model_name=model_name, output_size=6)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction='mean')
sch_dict = {
    'step': StepLR(optimizer, step_size=10, gamma=0.5),
    'cos': CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.1),
    'none': None
}
scheduler = sch_dict[arges.scheduler]

# Train  ################################################################
trainer = Trainer(dataloaders, net, device, num_epochs, criterion, optimizer, scheduler,
                  batch_multiplier=1, exp=exp_name)
trainer.train()
