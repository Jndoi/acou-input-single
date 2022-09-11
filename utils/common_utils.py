"""
@Project :acouinput_python
@File ：common_utils.py
@Date ： 2022/4/21 15:44
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import numpy as np
from scipy.signal import savgol_filter

from utils.plot_utils import show_d_cir, show_signals
from constants.constants import TAP_SIZE, LABEL_CLASSES, DataType, SignalPaddingTypeMap, SignalPaddingType, WINDOW_SIZE
import torch


def smooth_data(data, win_length=11, poly_order=2):
    data = savgol_filter(data, window_length=win_length, polyorder=poly_order, mode="nearest")
    return data


def is_static(one_step_data, threshold):  # 每次传入(20, 120)的数据
    one_step_data_std = smooth_data(np.std(one_step_data, axis=0))
    one_step_data_energy = smooth_data(np.sum(one_step_data, axis=0))
    show_signals(one_step_data_std)
    show_signals(one_step_data_energy)
    # data_abs_append_std = cls.smooth_data(data_abs_append.std(axis=0))
    # data_abs_append_std_sum = data_abs_append_std
    # data_abs_append_std_sum = np.sum(data_abs_append_std.reshape((5, -1), order='F'), axis=0)
    # data_std = np.sum(data_std.reshape((5, -1), order='F'), axis=0)
    # if sum(data_std <= threshold) > 2:
    #     return False
    # else:
    #     return True


def padding_signals(data, data_type, target_frames_num):
    # data_type: DataType.AbsDCir, DataType.RealPhase
    # show_d_cir(data.reshape(-1, tap_size, window_size))

    if data.shape[0] >= target_frames_num:  # need not padding
        return data
    if data_type == DataType.AbsDCir:
        data = np.r_[data, np.zeros((target_frames_num - data.shape[0], data.shape[1], data.shape[2], data.shape[3]),
                                    dtype=np.uint8)]
    elif data_type == DataType.RealPhase:
        last_phase = data[-1, -1]  # find the last value
        data = np.r_[data, np.ones((target_frames_num - data.shape[0], data.shape[1]),
                                   dtype=np.int16) * last_phase]
    else:
        raise Exception("padding method and data type don\'t match")
    return data


def padding_batch_signals(data, data_type):
    # 1. find the max length of signals in a batch
    max_len = 0
    for item in data:
        max_len = max(item[0].shape[0], max_len)
    batch_size = len(data)
    # 2. padding d cir data and phase data
    d_cir_x = []
    y = []
    if data_type == DataType.AbsDCir:
        for i in range(0, batch_size):
            d_cir_x.append(padding_signals(data[i][0], DataType.AbsDCir, max_len))
            y.append(data[i][1])
        return torch.tensor(np.array(d_cir_x)), torch.tensor(np.array(y))
    elif data_type == DataType.AbsDCirAndRealPhase:
        phase_x = []
        for i in range(0, batch_size):
            d_cir_x.append(padding_signals(data[i][0], DataType.AbsDCir, max_len))
            phase_x.append(padding_signals(data[i][1], DataType.RealPhase, max_len))
            y.append(data[i][2])
        return torch.tensor(np.array(d_cir_x)), torch.tensor(np.array(phase_x)), torch.tensor(np.array(y))
    else:
        raise Exception("Data Type error")


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
