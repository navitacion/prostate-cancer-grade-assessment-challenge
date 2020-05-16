import os
import gc
import time
import datetime
import torch
from torch import nn
from tensorboardX import SummaryWriter


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

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
                        val_i += 1

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


class Trainer_2:
    def __init__(self, dataloaders, net, device, num_epochs, optimizer, scheduler=None,
                 exp='exp_name', save_weight_path='../weights', limit=2000):
        self.dataloaders = dataloaders
        self.net = net.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp = exp
        self.writer = SummaryWriter(f'../tensorboard/{exp}')
        self.save_weight_path = save_weight_path
        self.limit = limit

    def train(self):

        print('PANDA Challenge Training Model V2')
        train_i, val_i = 0, 0
        best_loss = 1e+9
        best_weights = None
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            print('#'*30)
            t = time.time()

            for phase in ['train', 'val']:
                epoch_loss = 0

                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                for i, (imgs, score, grade, _) in enumerate(self.dataloaders[phase]):

                    # Training
                    with torch.set_grad_enabled(phase == 'train'):
                        self.optimizer.zero_grad()
                        imgs = imgs.to(self.device)
                        score = score.to(self.device)
                        grade = grade.to(self.device)

                        out1, out2 = self.net(imgs)

                        loss_1 = criterion_1(out1, score)
                        loss_2 = criterion_2(out2, grade.reshape((-1)))
                        # loss = loss_1 + loss_2

                        if phase == 'train':
                            loss_2.backward()
                            self.optimizer.step()

                    epoch_loss += loss_2.item() * imgs.size(0)

                    # Tensorboard
                    if phase == 'train':
                        # self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        self.writer.add_scalar(f'{phase}/loss_1', loss_1, train_i)
                        self.writer.add_scalar(f'{phase}/loss_2', loss_2, train_i)
                        train_i += 1
                    elif phase == 'val':
                        # self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
                        self.writer.add_scalar(f'{phase}/loss_1', loss_1, val_i)
                        self.writer.add_scalar(f'{phase}/loss_2', loss_2, val_i)
                        val_i += 1

                    # Memory Clear
                    del out1, out2, loss_1, loss_2
                    gc.collect()
                    torch.cuda.empty_cache()

                    # 1epochあたりの学習回数
                    if i == self.limit and phase == 'train':
                        break

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