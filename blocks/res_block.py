"""
@Project :acouinput_python
@File ：res_block.py
@Date ： 2022/4/13 14:14
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import torch.nn as nn
import torch.nn.functional as F
from blocks.convnet_utils import conv_bn, conv_bn_relu
from blocks.se_block import SEBlock
from blocks.ca_block import CABlock


class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 如果输入与输出的维度不同
        if stride != 1 or in_channels != out_channels:
            # add a 1x1 conv to make output be the same and the stride is same to the set stride
            self.shortcut = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                    stride=stride)
        # 如果输入与输出的维度相同
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                  stride=stride, padding=1)
        self.conv2 = conv_bn(in_channels=out_channels, out_channels=out_channels,
                             kernel_size=3, stride=1, padding=1)
        # self.se = SEBlock(out_channels, reduction=4)
        # self.ca = CABlock(out_channels, out_channels, reduction=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.dropout(out, 0.1) + self.shortcut(x)  # add ca attention and dropout
        # out = F.dropout(self.ca(out), 0.1) + self.shortcut(x)  # add ca attention and dropout
        # out = F.dropout(self.se(out), 0.1) + self.shortcut(x)  # add se attention and dropout
        # out = F.dropout(out, 0.1) + self.shortcut(x)  # add dropout
        # 在单个数据集的情况下CABlock(0.96/0.88)比SEBlock(0.97/0.83)的效果更好
        # 不加CABlock或SEBlock，(0.96/0.88)
        # 不加dropout，(0.96/0.84)
        out = F.relu6(out)
        return out
