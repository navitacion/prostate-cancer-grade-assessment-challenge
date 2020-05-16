import os
import glob
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import seed_everything, ImageTransform, PANDADataset, Trainer, get_dataloaders
from utils.image_transform import ImageTransform
from utils.dataset import PANDADataset, PANDADataset_2, PANDADataset_3
from model import Model_2


train = pd.read_csv('../data/input/train.csv')
data_dir = '../data/input/train_images'
transform = ImageTransform(224)


dataset = PANDADataset_3(train, data_dir, 'train', transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

print(len(dataloader.dataset))

for img, label in dataloader:

    print(img.size())
    print(label)

    break


