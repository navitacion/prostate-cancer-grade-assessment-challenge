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
from model import ModelEFN

pd.set_option('display.max.columns', None)
df = pd.read_csv('../data/grid_224_2/res.csv')
df_2 = pd.read_csv('../data/output/pred_res.csv')


for _ in range(10):
    i = random.randint(0, len(df))
    tar = df['image_id'].unique()[i]

    print(tar)
    print(df[df['image_id'] == tar])
    print(df_2[df_2['image_id'] == tar])


