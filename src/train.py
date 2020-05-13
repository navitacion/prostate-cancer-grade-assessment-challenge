import os
import glob
import random
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils import seed_everything, ImageTransform, PANDADataset, Trainer
from model import ModelEFN

# Config
train_size = 0.95
batch_size = 128
lr = 5e-4
num_epochs = 100
limit_per_epoch = 50
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

print('Data Num')
print('Train: ', len(dataloaders['train']))
print('Val: ', len(dataloaders['val']))

net = ModelEFN(output_size=4)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)


trainer = Trainer(dataloaders, net, device, num_epochs, criterion,
                  optimizer, scheduler, exp=exp_name, limit_per_epoch=limit_per_epoch)

trainer.train()


