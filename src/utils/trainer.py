import os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Trainer:

    def __init__(self, dataloaders, net, device, num_epochs, criterion, optimizer, schedular=None,
                 exp='exp_name', save_weight_path='../weights'):
        self.dataloaders = dataloaders
        self.net = net
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.exp = exp
        self.writer = SummaryWriter(f'../tensorboard/{exp}')

        self.net = self.net.to(self.device)
        self.save_weight_path = save_weight_path

    def train(self):

        print('PANDA Challenge Training Model')
        best_loss = 1e+9
        best_weights = None

        for epoch in self.num_epochs:
            print('#'*30)

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

                    epoch_loss += loss.item() * img.size()[0]

                    # Tensorboard
                    count = epoch * len(self.dataloaders[phase]) + i
                    self.writer.add_scalar(f'{phase}/batch_loss', loss, count)

                epoch_loss = epoch_loss / len(self.dataloaders[phase])
                self.writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch + 1)

            if self.schedular is not None:
                self.schedular.step()

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                filename = f'{self.exp}_epoch_{epoch+1}_loss_{best_loss:.3f}.pth'
                torch.save(self.net.state_dict(), os.path.join(self.save_weight_path, filename))
                best_weights = self.net.state_dict()

        self.writer.close()

        self.net.load_state_dict(best_weights)

        return self.net
