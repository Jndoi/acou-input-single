"""
@Project :acou-input-single
@File ：inception.py
@Date ： 2022/9/29 13:05
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=(1, 1), bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, bias=False),
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)  # 17 x 17 x 1024


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, 16, kernel_size=(1, 1))
        )
        self.branch5 = nn.Sequential(
            ConvBNReLU(in_channels, 16, kernel_size=(1, 1)),
            ConvBNReLU(16, 24, kernel_size=(5, 5), padding=(2, 2))
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, 16, kernel_size=(1, 1)),
            ConvBNReLU(16, 24, kernel_size=(3, 3), padding=(1, 1)),
            ConvBNReLU(24, 24, kernel_size=(3, 3), padding=(1, 1))
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ConvBNReLU(in_channels, 24, kernel_size=(1, 1))
        )

    def forward(self, x):
        # 1x1
        branch1 = self.branch1(x)
        # 1x1 -> 5x5
        branch5 = self.branch5(x)
        # 1x1 -> 3x3 -> 3x3
        branch3 = self.branch3(x)
        # avg pool -> 1x1
        branch_pool = self.branch_pool(x)

        return torch.cat((branch1, branch5, branch3, branch_pool), dim=1)

