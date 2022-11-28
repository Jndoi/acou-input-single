"""
@Project :acou-input-single
@File ：evaluation.py
@Date ： 2022/11/22 13:39
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from net import Net, BATCH_SIZE
from transceiver.receiver import Receiver
from utils.plot_utils import show_confusion_matrix
from utils.wav2pickle_utils import DataItem
from utils.common_utils import cal_cer_total
from utils.dataset_utils import get_data_loader
from sklearn.metrics import confusion_matrix, classification_report
from constants.constants import DatasetLoadType, START_INDEX_SHIFT, LabelVocabulary


def predict(base_path, filename):
    args = ["M", 32, "M", 64, "M", 128]
    net = Net(layers=args, in_channels=32, gru_input_size=128, gru_hidden_size=128,
              num_classes=26).cuda()
    # state_dict = torch.load('model/params_15epochs(16).pth')  # 2028 569
    state_dict = torch.load('model/params_10epochs.pth')  # 2028 569
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


def predict_real_time(base_path):
    args = ["M", 32, "M", 64, "M", 128]
    net = Net(layers=args, in_channels=32, gru_input_size=128, gru_hidden_size=128,
              num_classes=26).cuda()
    # state_dict = torch.load('model/params_15epochs(16).pth')  # 2028 569
    state_dict = torch.load('model/params_25epochs.pth')  # 2028 569
    net.load_state_dict(state_dict)
    net.eval()  # 禁用 dropout, 避免 BatchNormalization 重新计算均值和方差
    letter_dict = {}
    res = {}
    count = 0
    words_pre = []
    words_label = []
    with torch.no_grad():
        for root, dirs, files in os.walk(base_path):
            # for filename in tqdm(files):
            for filename in files:
                output_letter = ""
                split_d_cir = Receiver.receive_real_time(base_path, filename,
                                                         start_index_shift=START_INDEX_SHIFT,
                                                         augmentation_radio=None)
                for d_cir in split_d_cir:
                    d_cir = torch.tensor(d_cir).float() / 255
                    # show_d_cir(d_cir, is_frames=True)
                    d_cir = d_cir.unsqueeze(0)  # add batch_size dim: torch.Size([1, 4, 1, 60, 60])
                    output = net(d_cir.cuda())
                    predicted = torch.argmax(output, 0)
                    output_letter = output_letter + chr(predicted.cpu().numpy()+ord('a'))
                true_letter = filename.split("_")[0]
                if true_letter not in letter_dict:
                    letter_dict[true_letter] = 0
                    res[true_letter] = []
                res[true_letter].append(output_letter)
                if output_letter == true_letter:
                    count = count + 1
                    letter_dict[true_letter] = letter_dict[true_letter] + 1
                else:
                    pass
                    # print("------", filename)
                    # os.remove(os.path.join(base_path, filename))
                words_pre.append(output_letter)
                words_label.append(true_letter)
                print("{}: {}".format(true_letter, output_letter))
    # for i in res.items():
    #     print(i)
    print(cal_cer_total(words_pre, words_label))
    print(letter_dict)
    print(count)


def acoustic_input_evaluation():
    args = ["M", 32, "M", 64, "M", 128]
    net = Net(layers=args, in_channels=32, gru_input_size=128, gru_hidden_size=128,
              num_classes=26).cuda()
    state_dict = torch.load('model/params_20epochs.pth')
    net.load_state_dict(state_dict)
    net.eval()
    data_path = [
        r"data/dataset.pkl",
        r"data/dataset_10cm.pkl",
        r"data/dataset_20cm.pkl",
        r"data/dataset_five_fourth.pkl",
        r"data/dataset_four_fifth.pkl",
        r"data/dataset_single.pkl",
        r"data/dataset_10cm_single.pkl",
        r"data/dataset_20cm_single.pkl",
        r"data/dataset_five_fourth_single.pkl",
        r"data/dataset_four_fifth_single.pkl",
    ]
    _, _, test_loader = get_data_loader(loader_type=DatasetLoadType.UniformTrainValidAndTest,
                                        batch_size=BATCH_SIZE,
                                        data_path=data_path)
    y_pred = []
    y_true = []
    num = 0
    with torch.no_grad():
        for step, (d_cir_x_batch, y_batch) in enumerate(test_loader):
            d_cir_x_batch = d_cir_x_batch.float() / 255
            d_cir_x_batch = d_cir_x_batch.cuda()
            y_batch = y_batch.cuda().long()
            output = net(d_cir_x_batch)
            pred = torch.argmax(output, 1)
            y_batch = y_batch.cpu().numpy().tolist()
            pred = pred.cpu().numpy().tolist()
            num += len(pred)
            y_true.extend(y_batch)
            y_pred.extend(pred)
    # print(classification_report(y_pred, y_true))
    C = confusion_matrix(y_true, y_pred, labels=range(26))
    C = C.astype('float') / C.sum(axis=1)
    C = np.around(C, decimals=2)
    return C


def precision(C):
    # 预测为a中实际为a的比例
    return recall(C.T)


def recall(C):
    # 实际为a中预测为a的比例
    arr = []
    index = 0
    for predicted_letter_arr in C:
        arr.append(predicted_letter_arr[index]/np.sum(predicted_letter_arr))
        index = index + 1
    return np.array(arr)


def f1_score(precision_arr, recall_arr):
    return 2 * np.multiply(precision_arr, recall_arr) / (precision_arr + recall_arr)


if __name__ == '__main__':
    # pass
    # predict_real_time(r'D:\AcouInputDataSet\word') # checkpoint 75 8.43%
    predict_real_time(r'D:\AcouInputDataSet\tmp2')
    # show_confusion_matrix(get_confusion_matrix())
    # C = acoustic_input_evaluation()
    # precision_arr = precision(C)
    # recall_arr = recall(C)
    # f1_score_arr = f1_score(precision_arr, recall_arr)
    # print(precision_arr)
    # print(recall_arr)
    # print(f1_score_arr)
    # # ref: https://blog.csdn.net/mighty13/article/details/113873617
    # width = 0.25
    # plt.figure(figsize=(10, 6))
    # x = np.arange(len(LabelVocabulary))
    # # plt.bar(x=x - width, height=precision_arr, width=width, label='1')
    # # plt.bar(x=x, height=recall_arr, width=width, label='2')
    # # plt.bar(x=x + width, height=f1_score_arr, width=width, label='3')
    # plt.bar(x=x, height=precision_arr)
    # plt.xticks(x, labels=LabelVocabulary)
    # plt.ylabel('Percentage (%)')
    # plt.xlabel('Letter Type')
    # plt.margins(0, 0)
    # # plt.legend()
    # # plt.savefig()
    # plt.show()
