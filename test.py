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
from tensorboardX import SummaryWriter

from src.utils import seed_everything, ImageTransform, ImageTransform_2, ImageTransform_3, PANDADataset, Trainer, QWKLoss, Trainer_multifold, get_dataloaders
from src.model import ModelEFN, ModelEFN_2

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--expname')
parser.add_argument('-model', '--model_name', default='b0')
parser.add_argument('-trn_s', '--train_size', type=float, default=0.8)
parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('-lr', '--lr', type=float, default=0.0005)
parser.add_argument('-ims', '--image_size', type=int, default=256)
parser.add_argument('-img_n', '--img_num', type=int, default=36)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)
parser.add_argument('-fold', '--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
parser.add_argument('-sch', '--scheduler', choices=['step', 'cos', 'none', 'cos_2'], default='step')
parser.add_argument('--multi', action='store_true')

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
img_path = glob.glob('./data/grid_256_level_1/img/*.jpg')
# Background_rate = 0.2
# img_path = glob.glob('../data/grid_224_level_1/img/*.jpg')

# Labelデータの読み込み
# meta = pd.read_csv('./data/input/train.csv')
# meta = pd.read_csv('./data/input/modified_train.csv')   # 修正ver1
meta = pd.read_csv('./data/input/modified_train_v2.csv')  # 修正ver2  (score_3, 4, 5の割合を考慮)


# Data Augmentation
# transform = ImageTransform(config['img_size'])
# transform = ImageTransform_2(config['img_size'])  # cutout
transform = ImageTransform_3()  # Normalizeではなく255で割る


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
# multiがtrueの場合、すべてのfoldを使用。false（デフォルト）の場合は一つのfoldのみを使用
_multi = arges.multi
dataloaders = get_dataloaders(meta, arges.fold, img_path, transform, config, batch_size, multi=_multi)


for img, label in dataloaders['train']:
    print(img.size())
    print(label)
    break
