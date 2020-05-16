import os
import glob
import random
import argparse
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils import seed_everything, ImageTransform, PANDADataset, Trainer, get_dataloaders
from model import ModelEFN


# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname')
parser.add_argument('-model', '--model_name', default='b0')
parser.add_argument('-trn_s', '--train_size', type=float, default=0.95)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr', '--lr', type=float, default=0.0005)
parser.add_argument('-ims', '--image_size', type=int, default=224)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)

arges = parser.parse_args()


# Config
train_size = arges.train_size
batch_size = arges.batch_size
lr = arges.lr
num_epochs = arges.num_epoch
img_size = arges.image_size
seed = 42
exp_name = arges.expname

seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Loading
data_dir = '../data/input/train_images'
meta = pd.read_csv('../data/input/train.csv')
transform = ImageTransform(img_size)
meta = meta.sample(frac=1.0).reset_index(drop=True)
train = meta.iloc[:int(len(meta) * train_size)]
val = meta.iloc[int(len(meta) * train_size):]
print(len(train))
print(len(val))
del meta

train_dataset = PANDADataset(train, data_dir, 'train', transform)
val_dataset = PANDADataset(val, data_dir, 'val', transform)
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

print('Data Num')
print('Train: ', len(dataloaders['train']))
print('Val: ', len(dataloaders['val']))

net = ModelEFN(model_name=f'efficientnet-{arges.model_name}', output_size=6)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction='mean')
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

trainer = Trainer(dataloaders, net, device, num_epochs, criterion, optimizer, scheduler, exp=exp_name)

trainer.train()


