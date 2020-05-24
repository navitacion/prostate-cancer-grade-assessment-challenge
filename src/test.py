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

print([f'score_{v}' for v in np.arange(6)])

# transform = ImageTransform(img_size=128)
# dataset = PANDADataset_2(data_dir='../data/grid_224_2', transform=transform, phase='train')
#
# img, label = dataset.__getitem__(8)
#
# print(img.size())
# print(label)
# print(label.dtype)
#
# print(len(dataset))

# target_id = res.iloc[154]['image_id']
# img = Image.open(f'../data/grid_224_2/{target_id}.jpg')
# plt.imshow(img)
# plt.show()
#
# plt.imshow(255 - img)
# plt.show()

# train = pd.read_csv('../data/input/train.csv')
# data_dir = '../data/input/train_images'
# transform = ImageTransform(224)
#
#
# dataset = PANDADataset(train, data_dir, 'train', transform, use_tile=False, img_size=224, tiff_level=-1, img_num=12)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# model = ModelEFN()
#
# print(len(dataloader.dataset))
#
# img, label = dataset.__getitem__(0)
# print(img.size())
# print(label)
# print(img.max())
# print(img.min())
# print(img.mean())
# print(img.std())
#
#
# for img, label in dataloader:
#
#     print(img.size())
#
#     img = img[0].permute(2, 1, 0)
#     img = img * 255
#     plt.imshow(img)
#     plt.show()
#
#     break


