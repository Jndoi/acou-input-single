"""
@Project :acouinput_python
@File ：common_utils.py
@Date ： 2022/4/21 15:44
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import numpy as np
from utils.plot_utils import show_d_cir, show_signals
from constants.constants import TAP_SIZE, LABEL_CLASSES, DataType, SignalPaddingTypeMap, SignalPaddingType, WINDOW_SIZE
import torch


def is_static(data_std, threshold):  # 每次传入30
    # nums = data_abs.shape[1] // 30
    # data_abs_append = np.c_[data_abs, np.zeros((120, (nums+1) * 30 - data_abs.shape[1]))]
    # data_abs_append_std = cls.smooth_data(data_abs_append.std(axis=0))
    # data_abs_append_std_sum = data_abs_append_std
    # data_abs_append_std_sum = np.sum(data_abs_append_std.reshape((5, -1), order='F'), axis=0)
    data_std = np.sum(data_std.reshape((5, -1), order='F'), axis=0)
    # 一共包含6个分段
    if sum(data_std <= threshold) >= 4:
        return False
    else:
        return True


def padding_signals(data, data_type, target_frames_num):
    # data_type: DataType.AbsDCir, DataType.RealPhase
    # show_d_cir(data.reshape(-1, tap_size, window_size))
    if data.shape[0] >= target_frames_num:  # need not padding
        return data
    padding_method = SignalPaddingTypeMap.get(data_type)
    if padding_method == SignalPaddingType.ZeroPadding and data_type == DataType.AbsDCir:
        data = np.r_[data, np.zeros((target_frames_num - data.shape[0], data.shape[1], data.shape[2], data.shape[3]), dtype=np.uint8)]
    elif padding_method == SignalPaddingType.LastValuePadding and data_type == DataType.RealPhase:
        data = np.r_[data, np.zeros((target_frames_num - data.shape[0], data.shape[1]), dtype=np.int16)]
    else:
        raise Exception("padding_method and data_type don\'t match")
    return data


def padding_batch_signals(data):
    # 1. find the max length of signals in a batch
    max_len = 0
    for item in data:
        max_len = max(item[0].shape[0], max_len)
    batch_size = len(data)
    # 2. padding d cir data and phase data
    d_cir_x = []
    # phase_x = []
    y = []
    for i in range(0, batch_size):
        d_cir_x.append(padding_signals(data[i][0], DataType.AbsDCir, max_len))
        # phase_x.append(padding_signals(data[i][1], DataType.RealPhase, max_len))
        y.append(data[i][1])
    # return torch.tensor(np.array(d_cir_x)), torch.tensor(np.array(phase_x)), torch.tensor(np.array(y))
    return torch.tensor(np.array(d_cir_x)), torch.tensor(np.array(y))


def decode_labels(label):
    label = label.detach().cpu().numpy()  # pred: N T
    texts = []
    for seq in label:  # shape of label: batch_size, sequence_length
        string = ""
        for seq_item in seq:
            string += LABEL_CLASSES[seq_item]
        texts.append(''.join(string))
    return texts


def decode_predictions(pred):
    # ref: https://github.com/GabrielDornelles/TorchNN-OCR/blob/main/train.py
    pred = pred.detach().cpu().numpy()  # pred: N T
    texts = []
    for seq in pred:  # pred.shape[0]: batch_size
        # for each item in a batch, decode the ctc output
        string = ""
        for seq_item in seq:
            string += LABEL_CLASSES[seq_item]
        # change the class index to character
        # [h, h, e, l, l, l, o] -> [helo]
        # [h, h, e, l, -, l, o] -> [hello]
        string = string.split(LABEL_CLASSES[0])
        for k in range(len(string)):
            frame = string[k]
            if len(frame) > 1:
                curr_char = frame[0]
                frame_str = curr_char
                for c in frame[1:]:
                    if c == curr_char:
                        continue
                    else:
                        curr_char = c
                        frame_str = frame_str + curr_char
                string[k] = frame_str
        texts.append(''.join(string))
    return texts


if __name__ == '__main__':
    print(decode_predictions(torch.tensor([[0, 0, 2, 3, 3, 3]]).cuda()))
