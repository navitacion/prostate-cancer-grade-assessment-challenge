import os
import gc
import time
import cv2
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import torch
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'


class Trainer:
    """
    PANDA Competitionの学習用クラス
    """
    def __init__(self, dataloaders, net, device, num_epochs, criterion, optimizer, scheduler=None,
                 batch_multiplier=1, exp='exp_name', save_weight_path='../weights'):
        """
        :param dataloaders: dict
            データローダを辞書型に格納したもの
            'train': train_dataloader, 'val': val_dataloader
        :param net: torch.nn.Module
            モデル
        :param device: torch.device
            デバイス（CPU, GPU）
        :param num_epochs: int
            エポック数
        :param criterion: torch.nn.Module
            損失関数
        :param optimizer: torch.optim
            オプティマイザ
        :param scheduler: torch.optim.lr_scheduler
            スケジューラー
        :param batch_multiplier: int
            multiple minibatch
            3: 3バッチごとにパラメータを更新
        :param exp: str
            学習テスト名
        :param save_weight_path: str
            モデルの重みの保存先パス
        """
        self.dataloaders = dataloaders
        self.net = net
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_multiplier = batch_multiplier
        self.exp = exp
        self.writer = SummaryWriter(f'../tensorboard/{exp}')

        self.net = self.net.to(self.device)
        self.save_weight_path = save_weight_path

    def train(self):
        """
        学習を実行する。
        :return: self.net
            学習済みモデル
        """

        print('PANDA Challenge Training Model')
        train_i, val_i = 0, 0
        best_loss = 1e+9
        best_score = 0
        best_weights = None
        count = 0

        for epoch in range(self.num_epochs):
            print('#'*30)
            t = time.time()

            for phase in ['train', 'val']:
                epoch_loss = 0
                epoch_score = 0
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                for i, (img, label) in enumerate(self.dataloaders[phase]):
                    if img.size()[0] == 1:
                        continue

                    img = img.to(self.device)
                    label = label.to(self.device)

                    # batch_multiplierの回数後にパラメータを更新する
                    # Multiple minibatch
                    if (phase == 'train') and (count == 0):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        count = self.batch_multiplier

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = self.net(img)
                        loss = self.criterion(pred, label) / self.batch_multiplier
                        if phase == 'train':
                            loss.backward()
                            count -= 1

                    epoch_loss += loss.item() * img.size(0) * self.batch_multiplier

                    # Cohen Kappa
                    _, pred = torch.max(pred, 1)
                    score = cohen_kappa_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                                              weights='quadratic')
                    epoch_score += score * img.size(0)

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss * self.batch_multiplier, train_i)
                        self.writer.add_scalar(f'{phase}/batch_kappa', score, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss * self.batch_multiplier, val_i)
                        self.writer.add_scalar(f'{phase}/batch_kappa', score, val_i)
                        val_i += 1

                # epochごとの平均誤差を算出
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                self.writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch + 1)
                epoch_score = epoch_score / len(self.dataloaders[phase].dataset)
                self.writer.add_scalar(f'{phase}/epoch_kappa', epoch_score, epoch + 1)

                print(f'Epoch {epoch+1}  {phase}  Loss: {epoch_loss:.3f}')

            if self.scheduler is not None:
                self.scheduler.step()

            if phase == 'val' and epoch_score > best_score:
                best_score = epoch_score
                filename = f'{self.exp}_epoch_{epoch+1}_kappa_{best_score:.3f}.pth'
                torch.save(self.net.state_dict(), os.path.join(self.save_weight_path, filename))
                best_weights = self.net.state_dict()

            elapsedtime = time.time() - t
            print(f'Elapsed Time: {str(datetime.timedelta(seconds=elapsedtime))}')

        self.writer.close()

        self.net.load_state_dict(best_weights)

        return self.net


class Trainer_2:
    """
        PANDA Competitionの学習用クラス
        分割済みの画像とマスククラスの割合を学習する
    """
    def __init__(self, dataloaders, net, device, num_epochs, optimizer, scheduler=None,
                 batch_multiplier=1, exp='exp_name', save_weight_path='../weights'):
        """
        :param dataloaders: dict
            データローダを辞書型に格納したもの
            'train': train_dataloader, 'val': val_dataloader
        :param net: torch.nn.Module
            モデル
        :param device: torch.device
            デバイス（CPU, GPU）
        :param num_epochs: int
            エポック数
        :param optimizer: torch.optim
            オプティマイザ
        :param scheduler: torch.optim.lr_scheduler
            スケジューラー
        :param exp: str
            学習テスト名
        :param save_weight_path: str
            モデルの重みの保存先パス
        """
        self.dataloaders = dataloaders
        self.net = net.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_multiplier = batch_multiplier
        self.exp = exp
        self.writer = SummaryWriter(f'../tensorboard_2/{exp}')
        self.save_weight_path = save_weight_path

    def train(self):
        """
        学習を実行する。
        :return: self.net
            学習済みモデル
        """

        print('PANDA Challenge Training Model V2')
        train_i, val_i = 0, 0
        best_loss = 1e+9
        best_weights = None
        criterion = nn.MSELoss()
        count = 0

        for epoch in range(self.num_epochs):
            print('#'*30)
            t = time.time()

            for phase in ['train', 'val']:
                epoch_loss = 0

                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                for i, (img, label) in enumerate(self.dataloaders[phase]):
                    if img.size()[0] == 1:
                        continue

                    img = img.to(self.device)
                    label = label.to(self.device)

                    # batch_multiplierの回数後にパラメータを更新する
                    # Multiple minibatch
                    if (phase == 'train') and (count == 0):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        count = self.batch_multiplier

                    # Training
                    with torch.set_grad_enabled(phase == 'train'):
                        out = self.net(img)
                        out = F.softmax(out, dim=1)
                        loss = criterion(out, label) / self.batch_multiplier

                        if phase == 'train':
                            loss.backward()
                            count -= 1

                    epoch_loss += loss.item() * img.size(0) * self.batch_multiplier

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss * self.batch_multiplier, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss * self.batch_multiplier, val_i)
                        val_i += 1

                    # Memory Clear
                    del img, label
                    torch.cuda.empty_cache()

                # epochごとの平均誤差を算出
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                self.writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch + 1)

                print(f'Epoch {epoch+1}  {phase}  Loss: {epoch_loss:.3f}')

            if self.scheduler is not None:
                self.scheduler.step()

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                filename = f'{self.exp}_epoch_{epoch+1}_loss_{best_loss:.3f}.pth'
                torch.save(self.net.state_dict(), os.path.join(self.save_weight_path, filename))
                best_weights = self.net.state_dict()

            elapsedtime = time.time() - t
            print(f'Elapsed Time: {str(datetime.timedelta(seconds=elapsedtime))}')

        self.writer.close()

        self.net.load_state_dict(best_weights)

        return self.net

    def evaluate(self, img_path, transform, model_path=None, output_dir='../data/output'):
        """
        学習済みモデルを元に予測結果を返す。
        image_idごとに画像をモデルに適用。score_0~5の6つの割合を計算

        :param img_path: list or ndarray
            画像の全パス
        :param transform: albumentation
            前処理。Datasetに使用したtransformをそのまま使用
        :param model_path: str
            学習済みモデルの重み格納先ファイルパス
        :param output_dir: str
            出力先ディレクトリ
        """

        print('Evaluate...')
        res = pd.DataFrame()

        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.net.eval()
        self.net = self.net.to(self.device)

        for img, img_id in tqdm(self.dataloaders['test']):
            img = img.to(self.device)

            with torch.no_grad():
                pred = 0
                img = img.to(self.device)

                for f in [
                    lambda x: x,
                    lambda x: x.flip(-1),
                    lambda x: x.flip(-2),
                    lambda x: x.flip(-1, -2),
                    lambda x: x.transpose(-1, -2),
                    lambda x: x.transpose(-1, -2).flip(-1),
                    lambda x: x.transpose(-1, -2).flip(-2),
                    lambda x: x.transpose(-1, -2).flip(-1, -2),
                ]:
                    pred += F.softmax(self.net(f(img.clone())), 1)

                pred = pred / 8
                pred = pred.detach().cpu().numpy().tolist()

                pred = pd.DataFrame(pred, columns=[f'score_{v}' for v in np.arange(6)])
                pred['image_id'] = img_id

                res = pd.concat([res, pred], axis=0, ignore_index=True)
                res.to_csv(os.path.join(output_dir, 'pred_res.csv'), index=False)
