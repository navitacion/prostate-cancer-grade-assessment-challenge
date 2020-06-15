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
from model import ModelEFN, ModelEFN_2
from sklearn.metrics import cohen_kappa_score

# pd.set_option('display.max.columns', None)
# df = pd.read_csv('../data/grid_224_2/res.csv')
# df_2 = pd.read_csv('../data/output/pred_res.csv')
#
#
# for _ in range(10):
#     i = random.randint(0, len(df))
#     tar = df['image_id'].unique()[i]
#
#     print(tar)
#     print(df[df['image_id'] == tar])
#     print(df_2[df_2['image_id'] == tar])


z = torch.randn(2, 3, 224, 224)

model = ModelEFN_2('efficientnet-b0')
out = model(z)

pred = out.sum(1).round().int()
print(pred)
print(pred.dtype)

_, pred_2 = torch.max(out, 1)
print(pred_2)
print(pred_2.dtype)


a = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9],
                 [0.4, 0.5, 0.2, 0.1, 0.0]])

print(a.size())

criterion = nn.BCEWithLogitsLoss()

loss = criterion(out, a)
print(loss)

PREDS = a.sum(1).round().int().detach().numpy()

print(PREDS)
