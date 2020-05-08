import os
import glob
import random
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.utils import seed_everything
from utils.image_transform import ImageTransform
from utils.dataset import PANDADataset
from .utils.trainer import Trainer

# Config
train_size = 0.8
batch_size = 128
lr = 1e-3
num_epochs = 100
data_dir = '../data/grid_224'
seed = 42
exp_name = 'test_01'

seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Loading
img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
random.shuffle(img_path)
train_img_path = img_path[:int(len(img_path) * train_size)]
val_img_path = img_path[int(len(img_path) * train_size):]
score_df = pd.read_csv(os.path.join(data_dir, 'res.csv'))

transform = ImageTransform()
train_dataset = PANDADataset(train_img_path, score_df, transform, phase='train')
val_dataset = PANDADataset(val_img_path, score_df, transform, phase='val')

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

net = None
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


trainer = Trainer(dataloaders, net, device, num_epochs, criterion,
                  optimizer, exp=exp_name)

trainer.train()


