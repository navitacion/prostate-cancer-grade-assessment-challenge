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

from utils import seed_everything, PANDADataset_3, Trainer_2, ImageTransform, ImageTransform_2
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
parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-lr', '--lr', type=float, default=0.001)
parser.add_argument('-ims', '--image_size', type=int, default=224)
parser.add_argument('-epoch', '--num_epoch', type=int, default=100)

arges = parser.parse_args()
torch.cuda.empty_cache()

# Config
model_path = '../weights/mask_b0_01_epoch_9_loss_0.016.pth'
num_epochs = arges.num_epoch
data_dir = '../data/grid_224_2'
seed = 42
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = ImageTransform_2(img_size=arges.image_size)
img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
test_dataset = PANDADataset_3(img_path, transform)
test_dataloader = DataLoader(test_dataset, batch_size=arges.batch_size)
dataloaders = {
    'test': test_dataloader
}

net = Model_V2(output_size=6)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

trainer = Trainer_2(dataloaders, net, device, num_epochs, optimizer=None, scheduler=None)
trainer.evaluate(model_path=model_path, output_dir='../data/output')
