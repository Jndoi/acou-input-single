"""
@Project :acou-input-torch
@File ：single_net.py
@Date ： 2022/7/15 20:18
@Author ： Qiuyang Zeng
@Software ：PyCharm
https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/blob/master/model/inceptionv4.py
"""
from torch import nn
import torch
from transceiver.receiver import Receiver
from utils.dataset_utils import get_data_loader
from constants.constants import DatasetLoadType, WINDOW_SIZE, TAP_SIZE, START_INDEX_SHIFT, LabelVocabulary
from utils.wav2pickle_utils import DataItem
from utils.plot_utils import show_d_cir
from blocks.res_block import ResBasicBlock
import datetime


BATCH_SIZE = 32
EPOCH = 15
LR = 1e-3


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.nonlinear(x)


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):  # out_channels
        super(Stem, self).__init__()
        # 5x5 conv s2 pooling=2
        self.conv2d_1a = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            Conv2d(out_channels, out_channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        # branch0: 1x1 conv s1 + 3x3 max_pool s2
        # branch1: 3x3 conv s2
        self.pool_2a = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        # self.mixed_2a_branch_0 = nn.Sequential(
        #     Conv2d(out_channels, out_channels // 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # )
        # self.mixed_2a_branch_1 = Conv2d(out_channels, out_channels // 2,
        #                                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.pool_2a(x)
        # x0 = self.mixed_2a_branch_0(x)  # 1x1 conv s1 -> max_pool
        # x1 = self.mixed_2a_branch_1(x)  # 3x3 conv s2
        # x = torch.cat((x0, x1), dim=1)  # batch_size, channels, width, height
        return x


class Inception_A(nn.Module):  # 15 * 10 * 64 -> 15 * 10 * 64

    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch_0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        self.branch_1 = nn.Sequential(
            Conv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.branch_3 = nn.Sequential(
            Conv2d(in_channels, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x):
        # avg pool -> 1x1
        x0 = self.branch_0(x)
        # 1x1
        x1 = self.branch_1(x)
        # 1x1 -> 5x5
        x2 = self.branch_2(x)
        # 1x1 -> 3x3 -> 3x3
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Reduction_A(nn.Module):  # 15 * 10 * 64 -> 8 * 6 * 128
    # 35 -> 17
    def __init__(self, in_channels, k=16, l=64, m=64, n=64):
        super(Reduction_A, self).__init__()
        self.branch_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.branch_1 = Conv2d(in_channels, n, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.branch_2 = nn.Sequential(
            Conv2d(in_channels, k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2d(k, l, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2d(l, m, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)


class Net(nn.Module):
    def __init__(self, in_channels, gru_input_size, gru_hidden_size, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_input_size = gru_input_size
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            Stem(1, self.in_channels),  # 60 * 40 * 1 -> 15 * 10 * in_channels(64)
            Inception_A(self.in_channels * 2),  # 15 * 10 * in_channels -> 15 * 10 * in_channels(64)
            Reduction_A(self.in_channels * 2),  # 15 * 10 * in_channels -> 7 * 5 * in_channels(128)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, num_layers=2)
        self.cls = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.gru_hidden_size * 2, self.num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):   # shape of x: (batch_size, sequence_length, features)
        x = x.transpose(0, 1)   # (sequence_length, batch_size, 1, H, W)
        conv_items = []
        for x_item in x:
            conv_item = self.conv(x_item).unsqueeze(0)
            conv_items.append(conv_item)    # shape of conv_item: (1, batch_size, features)
        x = torch.cat(conv_items, 0)    # shape of x: (sequence_length, batch_size, features)
        _, h_n = self.gru(x)    # shape of x: (sequence_length, batch_size, gru_hidden_size)
        # shape of h_n: (1, batch_size, gru_hidden_size)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.reshape(h_n.shape[0], -1)
        output = self.cls(h_n.squeeze(0))  # shape of x: (batch_size, num_classes)
        return output


def evaluate(data_loader, net, type, total):
    correct = 0
    net.eval()
    with torch.no_grad():
        for step, (d_cir_x_batch, y_batch) in enumerate(data_loader):
            d_cir_x_batch = d_cir_x_batch.cuda()
            y_batch = y_batch.cuda()
            d_cir_x_batch = d_cir_x_batch.float() / 255
            output = net(d_cir_x_batch)
            predicted = torch.argmax(output, 1)
            correct += sum(y_batch == predicted).item()
        print("{}: {}/{}, acc {}".format(type, correct, total, round(correct * 1.0 / total, 6)))


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def train():
    net = Net(in_channels=64, gru_input_size=256, gru_hidden_size=128, num_classes=26).cuda()
    print_model_parm_nums(net)
    data_path = [r"data/dataset_single_smooth_20_40.pkl"]
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    # state_dict = torch.load('single_net_params.pth')  # 2028 569
    # net.load_state_dict(state_dict)
    train_loader, valid_loader, test_loader = get_data_loader(loader_type=DatasetLoadType.TrainValidAndTest,
                                                              batch_size=BATCH_SIZE,
                                                              data_path=data_path)
    train_size = len(train_loader.dataset)
    valid_size = len(valid_loader.dataset)
    test_size = len(test_loader.dataset)
    print("Train on {} samples, validate on {} samples, test on {} samples".format(train_size, valid_size, test_size))
    batch_num = train_size // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_num * 10, gamma=0.33)

    for epoch in range(EPOCH):
        net.train()
        correct = 0
        epoch_loss = 0
        start_time = datetime.datetime.now()
        for step, (d_cir_x_batch, y_batch) in enumerate(train_loader):
            d_cir_x_batch = d_cir_x_batch.float() / 255
            d_cir_x_batch = d_cir_x_batch.cuda()
            y_batch = y_batch.cuda().long()
            output = net(d_cir_x_batch)
            loss = loss_func(output, y_batch)  # (N,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 这里不再用optimizer.step()
            epoch_loss += loss.item()
            predicted = torch.argmax(output, 1)
            correct += sum(y_batch == predicted).item()
        end_time = datetime.datetime.now()
        print("[epoch {}] {}s {} acc {} loss {}".format(epoch + 1, (end_time - start_time).seconds, correct,
                                                        round(correct*1.0/train_size, 4), round(epoch_loss, 2)))
        if (epoch + 1) % 5 == 0:
            print("train loss: {} train acc: {}".format(round(epoch_loss, 2), round(correct*1.0/train_size, 4)))
            evaluate(valid_loader, net, "valid", valid_size)
            evaluate(test_loader, net, "test", test_size)
    torch.save(net.state_dict(), 'model/single_net_params_data_augmentation.pth')


def predict(base_path, filename):
    net = Net(in_channels=64, gru_input_size=128, gru_hidden_size=64, num_classes=26).cuda()
    state_dict = torch.load('model/single_net_params_data_augmentation.pth')  # 2028 569
    # state_dict = torch.load('single_net_params.pth')  # 2028 569
    net.load_state_dict(state_dict)
    if net is None or base_path is None or filename is None:
        raise Exception("please provide parameters")
    net.eval()  # 禁用 dropout, 避免 BatchNormalization 重新计算均值和方差
    arr = []
    char_dict = {}
    with torch.no_grad():
        for file in filename:
            label = file.split("_")[0]
            if label not in char_dict:
                char_dict[label] = []
            split_d_cir = Receiver.receive(base_path, file, gen_img=False,
                                           start_index_shift=START_INDEX_SHIFT,
                                           augmentation_radio=None)
            # sequence_length, tap_size, window_size
            # show_d_cir(split_d_cir, is_frames=True)
            split_d_cir = torch.tensor(split_d_cir).float() / 255
            split_d_cir = split_d_cir.unsqueeze(0)  # add batch_size dim: torch.Size([1, 4, 1, 60, 60])
            output = net(split_d_cir.cuda())
            predicted = torch.argmax(output, 0)
            arr.append(predicted.cpu().numpy())
            char_dict[label].append(chr(predicted.cpu().numpy()+ord('a')))
            # print("{} {}".format(file, predicted.data))
        total = 0
        correct = 0
        res = []
        for key, value in char_dict.items():
            item_total = len(value)
            item_correct = value.count(key)
            res.append(item_correct/item_total)
            print("{} {} {}".format(key, value, item_correct/item_total))
            total += item_total
            correct += item_correct
        print("acc:{}".format(correct/total))
        print(res)


if __name__ == '__main__':
    train()
    # import os
    # files = os.listdir(r"D:\AcouInputDataSet\single_test")
    # predict(r"D:\AcouInputDataSet\single_test", files)
