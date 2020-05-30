import os
import gc
import glob
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from utils.preprocessing import PANDAImagePreprocessing_2

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'


# radboudのマスク画像を対象にタイル分割せずマスク値のピクセル割合を計算する
# ラベル割合とisup_gradeがきちんと整合取れているか確認するために実施

# Config
data_dir = '../data/input'
save_dir = f'../data/'

# データ読み込み
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# 対象のデータを絞り込み
# maskデータがあるimage_idを抽出
masks = glob.glob(os.path.join(data_dir, 'train_label_masks', '*.tiff'))
masks = [id.split(sep)[-1].split('_')[0] for id in masks]
train = train[train['image_id'].isin(masks)].reset_index(drop=True)
train = train[train['data_provider'] == 'radboud'].reset_index(drop=True)

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

        prep = PANDAImagePreprocessing_2(target_id=id,
                                         tiff_level=0,
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

all_res.to_csv(os.path.join(save_dir, 'res_score_from_level_0.csv'), index=False)
