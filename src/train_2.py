import os
import gc
import glob
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils import seed_everything, PANDADataset_2, Trainer_2, ImageTransform, ImageTransform_2
from model import ModelEFN, Model_V2

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname', default='test_01_V2')
parser.add_argument('-model', '--model_name', default='b0')
parser.add_argument('-trn_s', '--train_size', type=float, default=0.8)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr', '--lr', type=float, default=0.001)
parser.add_argument('-ims', '--image_size', type=int, default=224)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)

arges = parser.parse_args()
torch.cuda.empty_cache()

# Config
train_size = arges.train_size
batch_size = arges.batch_size
lr = arges.lr
num_epochs = arges.num_epoch
data_dir = '../data/grid_224_2'
seed = 42
exp_name = arges.expname
model_name = f'efficientnet-{arges.model_name}'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = ImageTransform_2(img_size=arges.image_size)
df = pd.read_csv(os.path.join(data_dir, 'res.csv'))
df = df.sample(frac=1.0).reset_index(drop=True)
train = df.iloc[:int(len(df) * train_size)]
val = df.iloc[int(len(df) * train_size):]

del df
gc.collect()

train_dataset = PANDADataset_2(data_dir, train, transform=transform, phase='train')
val_dataset = PANDADataset_2(data_dir, val, transform=transform, phase='val')

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

print('Data Num')
print('Train: ', len(dataloaders['train'].dataset))
print('Val: ', len(dataloaders['val'].dataset))

net = Model_V2(output_size=6)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

trainer = Trainer_2(dataloaders, net, device, num_epochs, optimizer, scheduler,
                    batch_multiplier=1, exp=exp_name)

trainer.train()
