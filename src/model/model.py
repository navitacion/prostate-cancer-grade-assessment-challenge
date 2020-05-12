import os

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class ModelEFN(nn.Module):
    def __init__(self, model_name='efficientnet-b0', output_size=1):
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

        return out1, out2

