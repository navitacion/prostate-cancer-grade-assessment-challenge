import os
import time
import datetime
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, dataloaders, net, device, num_epochs, criterion, optimizer, scheduler=None,
                 exp='exp_name', save_weight_path='../weights', limit_per_epoch=50):
        self.dataloaders = dataloaders
        self.net = net
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp = exp
        self.limit_per_epoch = limit_per_epoch
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

                        _, pred = self.net(img)

                        loss = self.criterion(pred, label)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    epoch_loss += loss.item() * img.size()[0]

                    # Tensorboard
                    if phase == 'train':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, train_i)
                        train_i += 1
                    elif phase == 'val':
                        self.writer.add_scalar(f'{phase}/batch_loss', loss, val_i)
                        val_i += 1

                    # 1 epochあたりの学習回数を指定
                    if i >= self.limit_per_epoch and phase == 'train':
                        break

                # epochごとの平均誤差を算出
                if phase == 'train':
                    epoch_loss = epoch_loss / self.limit_per_epoch
                else:
                    epoch_loss = epoch_loss / len(self.dataloaders[phase])
                self.writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch + 1)

                print(f'Epoch {epoch}  {phase}  Loss: {epoch_loss:.3f}')

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
