import os
import gc
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils


if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'


def pad_and_tile(img, img_size, img_num=12):
    """
    画像を複数のタイルに分割し、img_num分の画像を抽出し、再度結合する
    :param img: ndarray
        画像
    :param img_size: int
        タイルに分割するときの画像サイズ
        タイルは(3, img_size, img_size)に分割される
    :param img_num: int
        再構築する際に使用する画像数
    :return: ndarray
        再構築後の画像
    """
    # Padding
    H, W = img.shape[:2]
    pad_h = (img_size - H % img_size) % img_size
    pad_w = (img_size - W % img_size) % img_size

    padded_img = np.pad(
        img,
        pad_width=[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=0
    )

    new_H, new_W = padded_img.shape[:2]

    del img
    gc.collect()
    img_list = []

    for h in range(int(new_H / img_size)):
        for w in range(int(new_W / img_size)):
            # Trim
            _img = padded_img[h * img_size:(h + 1) * img_size,
                              w * img_size:(w + 1) * img_size, :]

            img_list.append(_img)

    if len(img_list) < img_num:
        while True:
            img_list.append(np.zeros((img_size, img_size, 3)))
            if len(img_list) == img_num:
                break

    flag = np.array([np.sum(img) for img in img_list])
    flag = np.argsort(flag)[::-1][:img_num]

    img_list = np.stack(img_list, axis=0)
    img_list = img_list[flag]

    return img_list


class PANDADataset(Dataset):
    """
    image_preprocessingでタイル状にした画像を読み込み、v,hconcatした画像を出力する
    image_idに応じて画像を抽出し、その画像からランダムに選ぶ
    """

    def __init__(self, img_path, df, phase='train', transform=None, img_num=16, img_size=224):
        self.img_path = img_path
        self.df = df
        self.transform = transform
        self.phase = phase
        self.img_num = img_num
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_row = self.df.iloc[idx]
        target_id = target_row['image_id']

        target_img_path = [path for path in self.img_path if target_id in path]
        img = []

        for path in target_img_path:
            _img = cv2.imread(path)
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            _img = 255 - _img

            img.append(_img)

        if len(img) < self.img_num:
            while True:
                img.append(np.zeros((self.img_size, self.img_size, 3)))
                if len(img) == self.img_num:
                    break

        img = np.stack(img, axis=0)

        # 背景が少ない画像をピックアップするようにする
        flag = np.array([np.sum(im) for im in img])
        flag = np.argsort(flag)[::-1][:self.img_num]
        img = img[flag]

        if self.img_num == 16:
            # 複数のタイルをつなぎ合わせて1枚の画像にする
            # (4, 4)
            img = cv2.hconcat([
                cv2.vconcat([img[0], img[1], img[2], img[3]]),
                cv2.vconcat([img[4], img[5], img[6], img[7]]),
                cv2.vconcat([img[8], img[9], img[10], img[11]]),
                cv2.vconcat([img[12], img[13], img[14], img[15]])
            ])

        elif self.img_num == 12:
            # (4, 3)
            img = cv2.hconcat([
                cv2.vconcat([img[0], img[1], img[2], img[3]]),
                cv2.vconcat([img[4], img[5], img[6], img[7]]),
                cv2.vconcat([img[8], img[9], img[10], img[11]])
            ])

        elif self.img_num == 25:
            # (5, 5)
            img = cv2.hconcat([
                cv2.vconcat([img[0], img[1], img[2], img[3], img[4]]),
                cv2.vconcat([img[5], img[6], img[7], img[8], img[9]]),
                cv2.vconcat([img[10], img[11], img[12], img[13], img[14]]),
                cv2.vconcat([img[15], img[16], img[17], img[18], img[19]]),
                cv2.vconcat([img[20], img[21], img[22], img[23], img[24]])
            ])

        elif self.img_num == 36:
            # (6, 6)
            img = cv2.hconcat([
                cv2.vconcat([img[0], img[1], img[2], img[3], img[4], img[5]]),
                cv2.vconcat([img[6], img[7], img[8], img[9], img[10], img[11]]),
                cv2.vconcat([img[12], img[13], img[14], img[15], img[16], img[17]]),
                cv2.vconcat([img[18], img[19], img[20], img[21], img[22], img[23]]),
                cv2.vconcat([img[24], img[25], img[26], img[27], img[28], img[29]]),
                cv2.vconcat([img[30], img[31], img[32], img[33], img[34], img[35]]),
            ])

        if self.transform is not None:
            img = self.transform(img, phase=self.phase)
        else:
            img = torch.tensor(img).permute(2, 0, 1)
        img = img.to(torch.float32)

        label = target_row['isup_grade']

        return img, label


class PANDADataset_2(Dataset):
    """
    PANDADatasetとほぼ同じ
    タイルごとの画像ひとつひとつにaugmentをかけている
    """

    def __init__(self, img_path, df, phase='train', transform=None, img_num=36, img_size=224):
        self.img_path = img_path
        self.df = df
        self.transform = transform
        self.phase = phase
        self.img_num = img_num
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target_row = self.df.iloc[idx]
        target_id = target_row['image_id']

        target_img_path = [path for path in self.img_path if target_id in path]
        img = []

        for path in target_img_path:
            _img = cv2.imread(path)
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            _img = 255 - _img

            img.append(_img)

        if len(img) < self.img_num:
            while True:
                img.append(np.zeros((self.img_size, self.img_size, 3)))
                if len(img) == self.img_num:
                    break

        img = np.stack(img, axis=0)

        # 背景が少ない画像をピックアップするようにする
        flag = np.array([np.sum(im) for im in img])
        flag = np.argsort(flag)[::-1][:self.img_num]
        img = img[flag]

        img_augmented = []
        for i in range(img.shape[0]):
            img_augmented.append(self.transform(img[i], phase=self.phase))

        img_augmented = vutils.make_grid(img_augmented, normalize=False, padding=0, nrow=int(np.sqrt(self.img_num)))

        label = target_row['isup_grade']

        return img_augmented, label


def get_dataloaders(meta, fold, img_path, transform, img_num, img_size, batch_size, multi=False):

    if not multi:
        train_idx = np.where((meta['fold'] != fold))[0]
        valid_idx = np.where((meta['fold'] == fold))[0]
        train = meta.iloc[train_idx]
        val = meta.iloc[valid_idx]
        del meta

        train_dataset = PANDADataset(img_path, train, 'train', transform, img_num, img_size)
        val_dataset = PANDADataset(img_path, val, 'val', transform, img_num, img_size)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        }

        print('Data Num')
        print('Train: ', len(dataloaders['train'].dataset))
        print('Val: ', len(dataloaders['val'].dataset))
        print('#' * 30)
        print('Iterarion')
        print('Train: ', len(dataloaders['train']))
        print('Val: ', len(dataloaders['val']))

        return dataloaders

    else:
        # すべてのfold分のdataloaderを作成
        dataloaders = {}
        for f in range(5):
            train_idx = np.where((meta['fold'] != f))[0]
            valid_idx = np.where((meta['fold'] == f))[0]
            train = meta.iloc[train_idx]
            val = meta.iloc[valid_idx]

            train_dataset = PANDADataset(img_path, train, 'train', transform, img_num, img_size)
            val_dataset = PANDADataset(img_path, val, 'val', transform, img_num, img_size)

            dataloaders[f'train_{f}'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataloaders[f'val_{f}'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print('Data Num')
        print('Train: ', len(dataloaders['train_0'].dataset))
        print('Val: ', len(dataloaders['val_0'].dataset))
        print('#'*30)
        print('Iterarion')
        print('Train: ', len(dataloaders['train_0']))
        print('Val: ', len(dataloaders['val_0']))

        return dataloaders
