import os
import gc
import glob
import random
import openslide
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
    画像とラベルを出力するデータセット
    use_tileで画像を分割するかどうかを指定できる
    """

    def __init__(self, meta, data_dir, phase='train', transform=None, tiff_level=-1, use_tile=False, img_size=224,
                 img_num=16, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), tile_img_size=224):
        """
        :param meta: dataframe
            train.csv
        :param data_dir: str
            画像が格納されているディレクトリパス
        :param phase: str
            学習用(train) or 検証用(val)
        :param transform:
            データ拡張、前処理
        :param tiff_level: int
            tiffを読み込む際に指定するレベル
            0：最も解像度が高いもの　-1:最も解像度が低いもの
        :param use_tile: bool
            タイルで分割して再構築する処理を加えるかどうか
        :param img_size: int
            出力する画像サイズ
        :param img_num:
            タイル分割する際に使用する画像数
            12: (4 * 3)  16: (4 * 4)枚の画像を結合する
        :param mean: tuple
            前処理の正規化する際に使用する平均
        :param std: tuple
            前処理の正規化する際に使用する標準偏差
        :param tile_img_size:
            タイル分割する際にタイルごとの画像サイズ
        """
        self.meta = meta
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform
        self.tiff_level = tiff_level
        self.use_tile = use_tile
        self.img_size = img_size
        self.tile_img_size = tile_img_size
        self.img_num = img_num
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        target_row = self.meta.iloc[idx]
        target_id = target_row['image_id']
        label = target_row['isup_grade']

        img_path = os.path.join(self.data_dir, f'{target_id}.tiff')
        slide = openslide.OpenSlide(img_path)
        if self.tiff_level == -1:
            img = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
        else:
            img = slide.read_region((0, 0), self.tiff_level, slide.level_dimensions[self.tiff_level])

        # PIL -> ndarray
        img = np.asarray(img)
        # RGBA -> RGB
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        slide.close()

        # 画像の数値変換
        img = 255.0 - img

        # タイル状にして複数の画像にする
        # タイルサイズをpad_and_tileの第2引数に取る
        if self.use_tile:
            img = pad_and_tile(img, self.tile_img_size, self.img_num)
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

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1)
        img = img.to(torch.float32)
        return img, label


class PANDADataset_2(Dataset):
    """
    前処理済みの画像とマスクスコアを出力する
    事前にrun_prep.pyでタイルごとの画像を保存しておく
    """

    def __init__(self, data_dir, df, transform=None, phase='train'):
        """
        :param data_dir: str
            分割した画像の保存先ディレクトリ
        :param df: dataframe
            res.csv  image_idごとにマスク値の割合を格納したdataframe
        :param transform:
            データ拡張、前処理
        :param phase: str
            学習用(train) or 検証用(val)
        """
        self.data_dir = data_dir
        self.img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.df = df
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 対象のimage_id
        target_row = self.df.iloc[idx]
        # target_idに該当する画像を抽出
        target_id = target_row['image_id']
        # Score_0~5まで取ってくる
        label = target_row[1:].values.tolist()
        label = np.array(label, dtype=np.float32)

        img = cv2.imread(os.path.join(self.data_dir, f'{target_id}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = 255 - img

        if self.transform is not None:
            img = self.transform(img, phase=self.phase)
        else:
            img = torch.tensor(img).permute(2, 0, 1)

        label = torch.from_numpy(label)

        return img, label


