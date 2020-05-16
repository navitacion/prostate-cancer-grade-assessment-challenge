import os, random, glob
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader

from .dataset import PANDADataset, PANDADataset_2


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(data_dir, train, train_size=0.8, batch_size=128, transform=None):
    # Data Loading
    img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
    random.shuffle(img_path)
    train_img_path = img_path[:int(len(img_path) * train_size)]
    val_img_path = img_path[int(len(img_path) * train_size):]
    score_df = pd.read_csv(os.path.join(data_dir, 'res.csv'))

    train_dataset = PANDADataset(train_img_path, score_df, train, transform, phase='train')
    val_dataset = PANDADataset(val_img_path, score_df, train, transform, phase='val')

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    return dataloaders


def get_dataloaders_2(data_dir, score_df, train, transform=None, train_size=0.8, batch_size=128, img_num_per_id=50):

    image_ids, c = np.unique([t.split('_')[0] for t in score_df['image_id']], return_counts=True)
    # get_image_numより少ない枚数のidは対象外とする
    image_ids = image_ids[c > img_num_per_id]
    random.shuffle(image_ids)
    train_image_id = image_ids[:int(len(image_ids) * train_size)]
    val_image_id = image_ids[int(len(image_ids) * train_size):]

    # キーを作成
    score_df['key'] = score_df['image_id'].apply(lambda x: x.split('_')[0])

    train_score_df = score_df[score_df['key'].isin(train_image_id)]
    val_score_df = score_df[score_df['key'].isin(val_image_id)]

    del train_score_df['key'], val_score_df['key']

    print('Train Image Num: ', len(train_image_id))
    print('Valid Image Num: ', len(val_image_id))

    train_dataset = PANDADataset_2(train_score_df, train, img_num_per_id=img_num_per_id, data_dir=data_dir,
                                   phase='train', transform=transform)
    val_dataset = PANDADataset_2(val_score_df, train, img_num_per_id=img_num_per_id, data_dir=data_dir,
                                 phase='val', transform=transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    }

    return dataloaders