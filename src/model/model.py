import os

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class ModelEFN(nn.Module):
    def __init__(self, model_name='efficientnet-b0', output_size=6):
        super(ModelEFN, self).__init__()
        self.base = EfficientNet.from_pretrained(model_name=model_name, num_classes=512)

        self.block = nn.Sequential(
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
        )

        self.last = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        out1 = self.base(x)
        out2 = self.block(out1)
        out2 = self.last(out2)

        return out2


class Model_V2(nn.Module):
    def __init__(self, output_size=6):
        super(Model_V2, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.SELU(inplace=True)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.SELU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=3)

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.SELU(inplace=True)
        )

        self.pool_2 = nn.MaxPool2d(kernel_size=3)

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.SELU(inplace=True)
        )

        self.g_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.block_5 = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(64),
            nn.SELU(inplace=True)
        )

        self.last = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.pool(x)
        x = self.block_3(x)
        x = self.pool_2(x)
        x = self.block_4(x)
        x = self.g_pool(x)

        x = x.view(x.size(0), -1)
        x = self.block_5(x)
        x = self.last(x)

        return x
