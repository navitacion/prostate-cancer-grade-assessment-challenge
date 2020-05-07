import os
import gc
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from utils.preprocessing import PANDAImagePreprocessing

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

# Config
data_dir = '../data/input'
save_dir = '../data/grid_224'
SIZE = 224
BACKGROUND = 0.7

# データ読み込み
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# 対象のデータを絞り込み
# maskデータがあるimage_idを抽出
masks = glob.glob(os.path.join(data_dir, 'train_label_masks', '*.tiff'))
masks = [id.split(sep)[-1].split('_')[0] for id in masks]
train = train[train['image_id'].isin(masks)].reset_index(drop=True)
# データ提供元を"radboud"のものだけ扱う
train = train[train['data_provider'] == 'radboud'].reset_index(drop=True)
ids = train['image_id'].values
del train, masks
gc.collect()


# 前処理の実行
print('PANDA Challenge - Image Preprocessing')
print('Target Data Num: ', ids.shape[0])

all_res = pd.DataFrame()

with redirect_stdout(open(os.devnull, 'w')):
    for id in tqdm(ids):
        prep = PANDAImagePreprocessing(target_id=id,
                                       img_size=SIZE,
                                       background_rate=BACKGROUND,
                                       save_dir=save_dir)

        res = prep.transform()
        all_res = pd.concat([all_res, res], axis=0, ignore_index=True)
        all_res.to_csv(os.path.join(save_dir, 'res.csv'), index=False)

