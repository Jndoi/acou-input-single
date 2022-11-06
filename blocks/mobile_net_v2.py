"""
@Project :acouinput_python
@File ：mobile_net_v2.py
@Date ： 2022/5/16 16:21
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from torch import nn
import math


def conv_bn(inp, oup, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class BNeck(nn.Module):
    # t: expansion_radio
    # c: out_channels
    # n:
    # s: stride
    def __init__(self, in_channels, expansion_radio, out_channels, stride):
        super(BNeck, self).__init__()
        self.stride = stride
        expansion = int(in_channels * expansion_radio)
        # 1. expansion layer: Conv 1x1, ReLU6
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expansion, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(inplace=True)
        )
        # 2. depth-wise conv layer: Dwise 3x3, ReLU6
        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(expansion, expansion, kernel_size=(3, 3), stride=stride, padding=(1, 1),
                      groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(inplace=True)
        )
        # 3. point-wise conv layer: conv 1x1, Linear
        self.point_wise_conv = nn.Sequential(
            nn.Conv2d(expansion, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 4. residual shortcut
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.expand(x)
        out = self.depth_wise_conv(out)
        out = self.point_wise_conv(out)
        out = out + self.shortcut(x) if self.stride in (1, (1, 1)) else out
        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.block1 = conv_bn(3, 32, 3, 2, 1)  # inp, oup, kernel_size, stride, padding
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        self.bneck = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.bneck.append(BNeck(input_channel, t, output_channel, s))
                else:
                    self.bneck.append(BNeck(input_channel, t, output_channel, 1))
                input_channel = output_channel
        self.bneck = nn.Sequential(*self.bneck)
        self.block2 = conv_bn(320, 1280, 1, 1, 0)
        self.block3 = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.classifier = nn.Conv2d(1280, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self._initialize_weights()

    def forward(self, x):
        out = self.block1(x)
        out = self.bneck(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
