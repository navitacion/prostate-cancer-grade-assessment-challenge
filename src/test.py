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
from utils import seed_everything, ImageTransform, PANDADataset, Trainer
from utils.image_transform import ImageTransform
from utils.dataset import PANDADataset, PANDADataset_2
from model import Model_2, ModelEFN


train = pd.read_csv('../data/input/train.csv')
data_dir = '../data/input/train_images'
transform = ImageTransform(224)


dataset = PANDADataset(train, data_dir, 'train', transform, use_tile=False, img_size=224, tiff_level=-1, img_num=12)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = ModelEFN()

print(len(dataloader.dataset))

img, label = dataset.__getitem__(0)
print(img.size())
print(label)
print(img.max())
print(img.min())
print(img.mean())
print(img.std())


for img, label in dataloader:

    print(img.size())

    img = img[0].permute(2, 1, 0)
    img = img * 255
    plt.imshow(img)
    plt.show()

    break


