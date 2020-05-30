import os
import gc
import glob
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from utils.preprocessing import PANDAImagePreprocessing

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--img_size', type=int, default=224)
args = parser.parse_args()

# Config
SIZE = args.img_size
data_dir = '../data/input'
save_dir = f'../data/grid_{SIZE}_level_1/img'
save_dir_mask = f'../data/grid_{SIZE}_level_1/mask'
BACKGROUND = 0.2

# データ読み込み
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# 対象のデータを絞り込み
# maskデータがあるimage_idを抽出
masks = glob.glob(os.path.join(data_dir, 'train_label_masks', '*.tiff'))
masks = [id.split(sep)[-1].split('_')[0] for id in masks]
train = train[train['image_id'].isin(masks)].reset_index(drop=True)

del masks
gc.collect()

# 前処理の実行
print('PANDA Challenge - Image Preprocessing')
print('Target Data Num: ', len(train))

img_id_list = []
score_0, score_1, score_2 = [], [], []
score_3, score_4, score_5 = [], [], []

with redirect_stdout(open(os.devnull, 'w')):
    for i in tqdm(range(len(train))):

        id = train.iloc[i]['image_id']
        data_provider = train.iloc[i]['data_provider']

        prep = PANDAImagePreprocessing(target_id=id,
                                       img_size=SIZE,
                                       background_rate=BACKGROUND,
                                       save_dir=save_dir,
                                       save_dir_mask=save_dir_mask,
                                       tiff_level=1,
                                       data_provider=data_provider)

        res = prep.transform()
        if res is None:
            continue

        img_id_list.extend(res['image_id'].values.tolist())
        score_0.extend(res['score_0'].values.tolist())
        score_1.extend(res['score_1'].values.tolist())
        score_2.extend(res['score_2'].values.tolist())
        score_3.extend(res['score_3'].values.tolist())
        score_4.extend(res['score_4'].values.tolist())
        score_5.extend(res['score_5'].values.tolist())

all_res = pd.DataFrame({
    'image_id': img_id_list,
    'score_0': score_0,
    'score_1': score_1,
    'score_2': score_2,
    'score_3': score_3,
    'score_4': score_4,
    'score_5': score_5
})

all_res.to_csv(os.path.join(save_dir, 'res.csv'), index=False)
