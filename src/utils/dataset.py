import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'


class PANDADataset(Dataset):

    def __init__(self, data_dir, transform=None):
        # 画像のパスを取得
        self.img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
        # id部分だけを抽出
        img_ids = [path.split(sep)[-1].split('.')[0] for path in self.img_path]
        self.score_df = pd.read_csv(os.path.join(data_dir, 'res.csv'))
        # 画像が存在する行だけ抽出
        self.score_df = self.score_df[self.score_df['image_id'].isin(img_ids)].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.score_df)

    def __getitem__(self, idx):
        target_img_path = self.img_path[idx]
        target_id = target_img_path.split(sep)[-1].split('.')[0]
        target_row = self.score_df[self.score_df['image_id'] == target_id]

        img = Image.open(target_img_path)

        if self.transform is not None:
            img = self.transform(img)

        label = (target_row['score_3'].values, target_row['score_4'].values, target_row['score_5'].values)
        label = torch.tensor(label)

        return img, label
