import os, random, glob
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader

from .dataset import PANDADataset

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(data_dir, train_size=0.8, batch_size=128, transform=None):
    # Data Loading
    img_path = glob.glob(os.path.join(data_dir, '*.jpg'))
    random.shuffle(img_path)
    train_img_path = img_path[:int(len(img_path) * train_size)]
    val_img_path = img_path[int(len(img_path) * train_size):]
    score_df = pd.read_csv(os.path.join(data_dir, 'res.csv'))

    train_dataset = PANDADataset(train_img_path, score_df, transform, phase='train')
    val_dataset = PANDADataset(val_img_path, score_df, transform, phase='val')

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    return dataloaders
