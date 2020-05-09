import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class MyEfficientNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyEfficientNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.base = EfficientNet.from_pretrained('efficientnet-b0', num_classes=output_size)

    def forward(self, x):
        x = self.base(x)
        return x
