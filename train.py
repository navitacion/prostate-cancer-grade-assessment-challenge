import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from tensorboardX import SummaryWriter
import hydra
from omegaconf import DictConfig
import mlflow

from src.utils import seed_everything, ImageTransform, ImageTransform_2, ImageTransform_3
from src.utils import Trainer, QWKLoss, Trainer_multifold, get_dataloaders
from src.model import ModelEFN, ModelEFN_2

if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'

seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra.main('config.yml')
def main(cfg: DictConfig):
    # Config  ################################################################
    IMAGE_NUM = cfg.data.image_num
    IMAGE_SIZE = cfg.data.image_size
    exp_name = cfg.data.exp
    model_name = f'efficientnet-{cfg.data.model_name}'
    BATCH_SIZE = cfg.training.batch_size
    lr = cfg.training.lr
    NUM_EPOCHS = cfg.training.num_epoch
    FOLD = cfg.training.fold
    SCHEDULER = cfg.training.scheduler

    # Chenge Current Dir  ################################################################
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    # Data Loading  ################################################################
    # Background_rate = 0.7
    img_path = glob.glob('./data/grid_256_level_1/img/*.jpg')

    # Labelデータの読み込み
    # meta = pd.read_csv('./data/input/train.csv')
    # meta = pd.read_csv('./data/input/modified_train.csv')   # 修正ver1
    meta = pd.read_csv('./data/input/modified_train_v2.csv')  # 修正ver2  (score_3, 4, 5の割合を考慮)

    # Data Augmentation
    # transform = ImageTransform(config['img_size'])
    # transform = ImageTransform_2(config['img_size'])  # cutout
    transform = ImageTransform_3()  # Normalizeではなく255で割る

    # idごとの画像数を抽出しimg_numより少ないimgは対象外にする
    img_id = [s.split(sep)[-1].split('_')[0] for s in img_path]
    u, count = np.unique(img_id, return_counts=True)
    img_id = u[count > int(IMAGE_NUM * 0.5)]
    meta = meta[meta['image_id'].isin(img_id)].reset_index(drop=True)

    # StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    meta['fold'] = -1
    for i, (trn_idx, val_idx) in enumerate(cv.split(meta, meta['isup_grade'])):
        meta.loc[val_idx, 'fold'] = i

    # Dataset, DataLoader  ################################################################
    # multiがtrueの場合、すべてのfoldを使用。false（デフォルト）の場合は一つのfoldのみを使用
    dataloaders = get_dataloaders(meta, FOLD, img_path, transform,
                                  IMAGE_NUM, IMAGE_SIZE, BATCH_SIZE, multi=cfg.training.multi_fold)

    # Model  ################################################################
    net = ModelEFN_2(model_name=model_name, output_size=6)

    # Set Weight
    # model_path = './weights/efn_b0_fromjpg_augtile_04_epoch_18_loss_1.191_kappa_0.716.pth'
    # net.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = QWKLoss()

    sch_dict = {
        'step': StepLR(optimizer, step_size=4, gamma=0.5),
        'cos': CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=lr * 0.1),
        'cos_2': CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0),
        'none': None
    }
    scheduler = sch_dict[SCHEDULER]

    # ML Flow  ###########################################################################
    experient_name = f'PANDA_{cfg.data.model_name}'
    mlflow.set_experiment(experient_name)

    with mlflow.start_run():
        # パラメータを記録
        for k, v in cfg.data.items():
            mlflow.log_param('data/' + str(k), v)
        for k, v in cfg.training.items():
            mlflow.log_param('training/' + str(k), v)

        # Train  ################################################################
        writer = SummaryWriter(f'./tensorboard/{exp_name}')
        if cfg.training.multi_fold:
            trainer = Trainer_multifold(dataloaders, net, device, NUM_EPOCHS, criterion, optimizer, scheduler,
                                        exp=exp_name, writer=writer, save_weight_path='./weights')
        else:
            trainer = Trainer(dataloaders, net, device, NUM_EPOCHS, criterion, optimizer, scheduler,
                              exp=exp_name, writer=writer, save_weight_path='./weights')
        trainer.train()


if __name__ == '__main__':
    main()
