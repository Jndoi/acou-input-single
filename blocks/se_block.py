"""
@Project :acouinput_python
@File ：se_block.py
@Date ： 2022/5/15 13:18
@Author ： Qiuyang Zeng
@Software ：PyCharm
code from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
"""
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16, dim="2d"):
        super(SEBlock, self).__init__()
        self.dim = dim
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.dim == "2d":
            b, c, _, _ = x.size()
            y = self.avg_pool2d(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
        elif self.dim == "1d":
            b, c, _ = x.size()
            y = self.avg_pool1d(x).view(b, c)
            y = self.fc(y).view(b, c, 1)
            return x * y.expand_as(x)
