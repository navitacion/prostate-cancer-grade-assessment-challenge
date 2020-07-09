import os
import gc
import time
import cv2
import random
import datetime
from sklearn.metrics import cohen_kappa_score
import torch
import mlflow


if os.name == 'nt':
    sep = '\\'
else:
    sep = '/'


class Trainer:
    """
    PANDA Competitionの学習用クラス
    """
    def __init__(self, dataloaders, net, device, num_epochs, criterion, optimizer, scheduler=None,
                 exp=None, writer=None, save_weight_path='../weights', binning=False):
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
        :param exp:
            学習試行名
        :param writer:
            tensorboard
        :param save_weight_path: str
            モデルの重みの保存先パス
        !:param binning: str
            ビニングラベルを設定
        """
        self.dataloaders = dataloaders
        self.net = net
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp = exp
        self.writer = writer

        self.net = self.net.to(self.device)
        self.save_weight_path = save_weight_path
        self.binning = binning

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

        for epoch in range(self.num_epochs):
            print('#'*30)
            t = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

            for phase in ['train', 'val']:

                label_list = []
                pred_list = []
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
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = self.net(img)
                        loss = self.criterion(pred, label)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    epoch_loss += loss.item() * img.size(0)

                    if self.binning:
                        # binning -> scalar
                        pred = pred.sigmoid().sum(1).round().int()
                        label = label.sum(1).round().int()
                    else:
                        _, pred = torch.max(pred, 1)

                    label_list.append(label)
                    pred_list.append(pred)

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
                        val_i += 1

                    # Memory Clear
                    del img, label, loss, pred
                    gc.collect()
                    torch.cuda.empty_cache()

                # epochごとの平均誤差を算出
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                self.writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch + 1)
                mlflow.log_metric(f'{phase}/epoch_loss', epoch_loss, epoch + 1)

                # Cohen Kappa
                label_list = torch.cat(label_list).cpu().numpy()
                pred_list = torch.cat(pred_list).cpu().numpy()
                acc = (pred_list == label_list).mean() * 100.
                epoch_score = cohen_kappa_score(label_list, pred_list, weights='quadratic')
                self.writer.add_scalar(f'{phase}/epoch_kappa', epoch_score, epoch + 1)
                self.writer.add_scalar(f'{phase}/epoch_acc', acc, epoch + 1)
                mlflow.log_metric(f'{phase}/epoch_kappa', epoch_score, epoch + 1)
                mlflow.log_metric(f'{phase}/epoch_acc', acc, epoch + 1)

                print(f'Epoch {epoch+1}  {phase}  Loss: {epoch_loss:.3f}')

                if phase == 'val' and (epoch_loss < best_loss or epoch_score > best_score):
                    best_loss = epoch_loss
                    best_score = epoch_score
                    filename = f'{self.exp}_epoch_{epoch+1}_loss_{best_loss:.3f}_kappa_{best_score:.3f}.pth'
                    torch.save(self.net.state_dict(), os.path.join(self.save_weight_path, filename))
                    mlflow.log_artifact(os.path.join(self.save_weight_path, filename))

            elapsedtime = time.time() - t
            print(f'Elapsed Time: {str(datetime.timedelta(seconds=elapsedtime))}')

        self.writer.close()


class Trainer_multifold:
    """
    PANDA Competitionの学習用クラス
    multifoldに対応させたもの
    """
    def __init__(self, dataloaders, net, device, num_epochs, criterion, optimizer, scheduler=None,
                 exp='exp_name', writer=None, save_weight_path='../weights', binning=False):
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
        :param exp: str
            学習テスト名
        :param save_weight_path: str
            モデルの重みの保存先パス
        !:param binning: str
            ビニングラベルを設定
        """
        self.dataloaders = dataloaders
        self.net = net
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp = exp
        self.writer = writer

        self.net = self.net.to(self.device)
        self.save_weight_path = save_weight_path
        self.binning = binning

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

        for epoch in range(self.num_epochs):
            print('#'*30)
            t = time.time()
            random.seed()
            fold = random.randint(0, 4)
            random.seed(42)

            for phase in ['train', 'val']:
                label_list = []
                pred_list = []
                epoch_loss = 0
                epoch_score = 0
                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                for i, (img, label) in enumerate(self.dataloaders[phase + f'_{fold}']):
                    if img.size()[0] == 1:
                        continue

                    img = img.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = self.net(img)
                        loss = self.criterion(pred, label)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    epoch_loss += loss.item() * img.size(0)

                    if self.binning:
                        # binning -> scalar
                        pred = pred.sigmoid().sum(1).round().int()
                        label = label.sum(1).round().int()
                    else:
                        _, pred = torch.max(pred, 1)

                    label_list.append(label)
                    pred_list.append(pred)

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
                        val_i += 1

                    # Memory Clear
                    del img, label, loss, pred
                    gc.collect()
                    torch.cuda.empty_cache()

                # epochごとの平均誤差を算出
                epoch_loss = epoch_loss / len(self.dataloaders[phase + f'_{fold}'].dataset)
                self.writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch + 1)
                mlflow.log_metric(f'{phase}/epoch_loss', epoch_loss, epoch + 1)

                # Cohen Kappa
                label_list = torch.cat(label_list).cpu().numpy()
                pred_list = torch.cat(pred_list).cpu().numpy()
                acc = (pred_list == label_list).mean() * 100.
                epoch_score = cohen_kappa_score(label_list, pred_list, weights='quadratic')
                self.writer.add_scalar(f'{phase}/epoch_kappa', epoch_score, epoch + 1)
                self.writer.add_scalar(f'{phase}/epoch_acc', acc, epoch + 1)
                mlflow.log_metric(f'{phase}/epoch_kappa', epoch_score, epoch + 1)
                mlflow.log_metric(f'{phase}/epoch_acc', acc, epoch + 1)

                print(f'Epoch {epoch+1}  {phase}  Loss: {epoch_loss:.3f}')

            if self.scheduler is not None:
                self.scheduler.step()

            if phase == 'val' and (epoch_loss < best_loss or epoch_score > best_score):
                best_loss = epoch_loss
                best_score = epoch_score
                filename = f'{self.exp}_epoch_{epoch+1}_loss_{best_loss:.3f}_kappa_{best_score:.3f}.pth'
                torch.save(self.net.state_dict(), os.path.join(self.save_weight_path, filename))
                mlflow.log_artifact(os.path.join(self.save_weight_path, filename))
                best_weights = self.net.state_dict()

            elapsedtime = time.time() - t
            print(f'Elapsed Time: {str(datetime.timedelta(seconds=elapsedtime))}')

        self.writer.close()

        self.net.load_state_dict(best_weights)

        return self.net
