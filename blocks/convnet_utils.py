"""
@Project :acouinput_python
@File ：convnet_utils.py
@Date ： 2022/4/13 14:15
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, nonlinear=None):  # nonlinear stands for activate function
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()  # just output the input without any operations
        else:
            self.nonlinear = nonlinear
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))  # with bn
        else:
            return self.nonlinear(self.conv(x))  # without bn


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, nonlinear=nn.ReLU())
