import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR

from utils import seed_everything, ImageTransform, ImageTransform_2, PANDADataset, Trainer, QWKLoss, Trainer_multifold
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
parser.add_argument('-fold', '--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
parser.add_argument('-sch', '--scheduler', choices=['step', 'cos', 'none', 'cos_2'], default='step')

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
# Background_rate = 0.7
img_path = glob.glob('../data/grid_224_2/*.jpg')
# Background_rate = 0.2
# img_path = glob.glob('../data/grid_224_level_1/img/*.jpg')

# Labelデータの読み込み
# meta = pd.read_csv('../data/input/train.csv')
# meta = pd.read_csv('../data/input/modified_train.csv')   # 修正ver1
meta = pd.read_csv('../data/input/modified_train_v2.csv')  # 修正ver2  (score_3, 4, 5の割合を考慮)


# Data Augmentation
transform = ImageTransform(config['img_size'])
# transform = ImageTransform_2(config['img_size'])  # cutout


# idごとの画像数を抽出しimg_numより少ないimgは対象外にする
img_id = [s.split(sep)[-1].split('_')[0] for s in img_path]
u, count = np.unique(img_id, return_counts=True)
img_id = u[count > int(config['img_num'] * 0.5)]
meta = meta[meta['image_id'].isin(img_id)].reset_index(drop=True)

# StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
meta['fold'] = -1
for i, (trn_idx, val_idx) in enumerate(cv.split(meta, meta['isup_grade'])):
    meta.loc[val_idx, 'fold'] = i


# Dataset, DataLoader  ################################################################
# すべてのfold分のdataloaderを作成
dataloaders = {}
for f in range(5):
    train_idx = np.where((meta['fold'] != f))[0]
    valid_idx = np.where((meta['fold'] == f))[0]
    train = meta.iloc[train_idx]
    val = meta.iloc[valid_idx]

    train_dataset = PANDADataset(img_path, train, 'train', transform, **config)
    val_dataset = PANDADataset(img_path, val, 'val', transform, **config)

    dataloaders[f'train_{f}'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloaders[f'val_{f}'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print('Data Num')
print('Train: ', len(dataloaders['train_0'].dataset))
print('Val: ', len(dataloaders['val_0'].dataset))
print('#'*30)
print('Iterarion')
print('Train: ', len(dataloaders['train_0']))
print('Val: ', len(dataloaders['val_0']))


# 単一のfoldのみで学習
# fold = arges.fold
# train_idx = np.where((meta['fold'] != fold))[0]
# valid_idx = np.where((meta['fold'] == fold))[0]
# train = meta.iloc[train_idx]
# val = meta.iloc[valid_idx]
# del meta
#
# train_dataset = PANDADataset_4(img_path, train, 'train', transform, **config)
# val_dataset = PANDADataset_4(img_path, val, 'val', transform, **config)
#
# dataloaders = {
#     'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
#     'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# }
#
# print('Data Num')
# print('Train: ', len(dataloaders['train'].dataset))
# print('Val: ', len(dataloaders['val'].dataset))
# print('#'*30)
# print('Iterarion')
# print('Train: ', len(dataloaders['train']))
# print('Val: ', len(dataloaders['val']))

# Model  ################################################################
net = ModelEFN(model_name=model_name, output_size=6)

# Set Weight
# model_path = '../weights/efn_b0_fromjpg_modify_03_01_epoch_3_loss_0.763_kappa_0.803.pth'
# net.load_state_dict(torch.load(model_path, map_location=device))

optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction='mean')
# criterion = QWKLoss()

sch_dict = {
    'step': StepLR(optimizer, step_size=10, gamma=0.5),
    'cos': CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.1),
    'cos_2': CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0),
    'none': None
}
scheduler = sch_dict[arges.scheduler]

# Train  ################################################################
trainer = Trainer_multifold(dataloaders, net, device, num_epochs, criterion, optimizer, scheduler,
                            batch_multiplier=1, exp=exp_name)
trainer.train()
