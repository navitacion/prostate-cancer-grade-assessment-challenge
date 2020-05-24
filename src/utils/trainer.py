import os
import gc
import time
import cv2
import datetime
import numpy as np
import pandas as pd
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
    def __init__(self, dataloaders, net, device, num_epochs, criterion, optimizer, scheduler=None,
                 exp='exp_name', save_weight_path='../weights'):
        self.dataloaders = dataloaders
        self.net = net
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp = exp
        self.writer = SummaryWriter(f'../tensorboard/{exp}')

        self.net = self.net.to(self.device)
        self.save_weight_path = save_weight_path

    def train(self):

        print('PANDA Challenge Training Model')
        train_i, val_i = 0, 0
        best_loss = 1e+9
        best_weights = None

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

                    self.optimizer.zero_grad()
                    img = img.to(self.device)
                    label = label.to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        pred = self.net(img)
                        loss = self.criterion(pred, label)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    epoch_loss += loss.item() * img.size(0)
                    # Cohen Kappa
                    _, pred = torch.max(pred, 1)
                    score = cohen_kappa_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                                              weights='quadratic')
                    epoch_score += score * img.size(0)

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        self.writer.add_scalar(f'{phase}/batch_kappa', score, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
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


class Trainer_2:
    def __init__(self, dataloaders, net, device, num_epochs, optimizer, scheduler=None,
                 exp='exp_name', save_weight_path='../weights'):
        self.dataloaders = dataloaders
        self.net = net.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp = exp
        self.writer = SummaryWriter(f'../tensorboard_2/{exp}')
        self.save_weight_path = save_weight_path

    def train(self):

        print('PANDA Challenge Training Model V2')
        train_i, val_i = 0, 0
        best_loss = 1e+9
        best_weights = None
        criterion = nn.MSELoss()

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

                    # Training
                    with torch.set_grad_enabled(phase == 'train'):
                        self.optimizer.zero_grad()
                        img = img.to(self.device)
                        label = label.to(self.device)

                        out = self.net(img)
                        out = F.softmax(out, dim=1)
                        loss = criterion(out, label)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    epoch_loss += loss.item() * img.size(0)

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
                        val_i += 1

                    # Memory Clear
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
        res = pd.DataFrame()

        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.net.eval()

        for path in img_path:
            img_id = path.split(sep)[-1].split('.')[0]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = 255 - img
            img = transform(img, phase='val')

            with torch.no_grad():
                pred = self.net(img)
                pred = F.softmax(pred, dim=1)
                pred = pred.detach().cpu().numpy().tolist()

                pred = pd.DataFrame(pred, columns=[f'score_{v}' for v in np.arange(6)])
                pred['image_id'] = img_id

                res = pd.concat([res, pred], axis=0, ignore_index=True)
                res.to_csv(os.path.join(output_dir, 'pred_res.csv'), index=False)
