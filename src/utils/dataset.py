import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'


class PANDADataset(Dataset):

    def __init__(self, img_path, score_df, transform=None, phase='train'):
        self.img_path = img_path
        # id部分だけを抽出
        img_ids = [path.split(sep)[-1].split('.')[0] for path in img_path]
        # 画像が存在する行だけ抽出
        self.score_df = score_df[score_df['image_id'].isin(img_ids)].reset_index(drop=True)
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.score_df)

    def __getitem__(self, idx):
        target_img_path = self.img_path[idx]
        target_id = target_img_path.split(sep)[-1].split('.')[0]
        target_row = self.score_df[self.score_df['image_id'] == target_id]

        # OpenCV
        img = cv2.imread(target_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 0.0 ~ 1.0
        img = img / 255

        if self.transform is not None:
            img = self.transform(img, phase=self.phase)

        label = (target_row['score_3'].values, target_row['score_4'].values, target_row['score_5'].values)
        label = torch.tensor(label)
        label = label.reshape((-1))

        return img, label
