"""
@Project :acou-input-torch
@File ：single_net.py
@Date ： 2022/7/15 20:18
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from torch import nn
import torch
from transceiver.receiver import Receiver
from utils.dataset_utils import get_data_loader
from constants.constants import DatasetLoadType, WINDOW_SIZE, TAP_SIZE, START_INDEX_SHIFT, LabelVocabulary
from utils.wav2pickle_utils import DataItemWithPhase
import datetime


BATCH_SIZE = 8
EPOCH = 100
LR = 1e-3


class Net(nn.Module):
    def __init__(self, gru_hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.gru_hidden_size = gru_hidden_size
        self.num_classes = num_classes
        self.phase_gru = nn.GRU(WINDOW_SIZE, self.gru_hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.phase_cls = nn.Sequential(
            nn.Linear(self.gru_hidden_size*num_layers, self.gru_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.gru_hidden_size, self.num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):   # shape of x: (batch_size, sequence_length, features)
        _, h_n = self.phase_gru(x)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.reshape(h_n.shape[0], -1)
        output = self.phase_cls(h_n.squeeze(0))
        return output


def evaluate(data_loader, net, type, total):
    correct = 0
    net.eval()
    with torch.no_grad():
        for step, (_,real_phase_x_batch, y_batch) in enumerate(data_loader):
            real_phase_x_batch = real_phase_x_batch.cuda()
            y_batch = y_batch.cuda()
            real_phase_x_batch = real_phase_x_batch.float() * 0.025
            output = net(real_phase_x_batch)
            predicted = torch.argmax(output, 1)
            correct += sum(y_batch == predicted).item()
        print("{}: {}/{}, acc {}".format(type, correct, total, round(correct * 1.0 / total, 6)))


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def train():
    # 0.902564
    # ["RES_32", "M", "RES_32",  "M", "RES_64", "M", "RES_64"]
    # 0.838462
    # [16, "M", 32,  "M", 48, "M", 64]
    # Conv2dWithBN(1, in_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    net = Net(gru_hidden_size=64, num_classes=26).cuda()
    print_model_parm_nums(net)
    data_path = [r"data/dataset_single_smooth.pkl", ]
    save = True
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.002)
    # state_dict = torch.load('single_net_params.pth')  # 2028 569
    # net.load_state_dict(state_dict)
    train_loader, valid_loader, test_loader = \
        get_data_loader(loader_type=DatasetLoadType.TrainValidAndTest, batch_size=BATCH_SIZE, data_path=data_path)
    train_size = len(train_loader.dataset)
    valid_size = len(valid_loader.dataset)
    test_size = len(test_loader.dataset)
    print("Train on {} samples, validate on {} samples, test on {} samples".format(train_size, valid_size, test_size))
    batch_num = train_size // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_num * 10, gamma=0.5)

    for epoch in range(EPOCH):
        net.train()
        correct = 0
        epoch_loss = 0
        start_time = datetime.datetime.now()
        for step, (_, real_phase_batch, y_batch) in enumerate(train_loader):
            real_phase_batch = real_phase_batch.float() * 0.025
            real_phase_batch = real_phase_batch.cuda()
            y_batch = y_batch.cuda().long()
            output = net(real_phase_batch)
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
    if save:
        torch.save(net.state_dict(), 'model/single_net_params_data_augmentation.pth')


def predict(base_path, filename):
    net = Net(gru_hidden_size=64, num_classes=26).cuda()
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
            predicted = torch.argmax(output, 1)
            arr.append(predicted.cpu().numpy()[0])
            char_dict[label].append(chr(predicted.cpu().numpy()[0]+ord('a')))
            # print("{} {}".format(file, predicted.data))
        total = 0
        correct = 0
        for key, value in char_dict.items():
            item_total = len(value)
            item_correct = value.count(key)
            print("{} {} {}".format(key, value, item_correct/item_total))
            total += item_total
            correct += item_correct
        print("acc:{}".format(correct/total))


# train loss: 68.18 train acc: 0.9847
# valid: 1813/1950, acc 0.929744
# test: 1812/1950, acc 0.929231
if __name__ == '__main__':
    train()
    # import os
    # files = os.listdir(r"D:\AcouInputDataSet\single_test")
    # predict(r"D:\AcouInputDataSet\single_test", files)
    # 0.777
