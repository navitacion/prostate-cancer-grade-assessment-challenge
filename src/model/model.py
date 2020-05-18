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


class Model_2(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(Model_2, self).__init__()
        self.base = EfficientNet.from_pretrained(model_name=model_name, num_classes=128)

        self.block = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
        )

        self.last_1 = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        b, n, c, w, h = x.size()
        x = x.view(-1, c, w, h)
        x = self.base(x)
        x = self.block(x)
        out1 = self.last_1(x)
        out1 = out1.view(b, n, -1)
        x = x.view(b, -1)
        out2 = Linear_custum(in_features=x.size(1))(x)

        return out1, out2


class Linear_custum(nn.Module):
    def __init__(self, in_features):
        super(Linear_custum, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=6).cuda()

    def forward(self, x):
        return self.linear(x)
