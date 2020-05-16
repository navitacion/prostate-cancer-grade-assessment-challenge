import os
import glob
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils import seed_everything, PANDADataset_2, Trainer, Trainer_2, get_dataloaders, ImageTransform, get_dataloaders_2
from model import ModelEFN, Model_2

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname', default='test_01_V2')
parser.add_argument('-trn_s', '--train_size', type=float, default=0.9)
parser.add_argument('-bs', '--batch_size', type=int, default=10)
parser.add_argument('-lr', '--lr', type=float, default=0.0005)
parser.add_argument('-ims', '--image_size', type=int, default=224)
parser.add_argument('-imn_id', '--image_num_per_id', type=int, default=15)
parser.add_argument('-lim', '--limit', type=int, default=50)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)

arges = parser.parse_args()
torch.cuda.empty_cache()

# Config
train_size = arges.train_size
batch_size = arges.batch_size
lr = arges.lr
num_epochs = arges.num_epoch
data_dir = '../data/grid_{}'.format(224)
seed = 42
img_num_per_id = arges.image_num_per_id
exp_name = arges.expname
limit = arges.limit
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = ImageTransform()
score_df = pd.read_csv(os.path.join(data_dir, 'res.csv'))
train = pd.read_csv('../data/input/train.csv')

dataloaders = get_dataloaders_2(data_dir, score_df, train, transform, train_size, batch_size, img_num_per_id)

print('Data Num')
print('Train: ', len(dataloaders['train']))
print('Val: ', len(dataloaders['val']))

net = Model_2()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

trainer = Trainer_2(dataloaders, net, device, num_epochs, optimizer, scheduler, exp=exp_name)

trainer.train()
