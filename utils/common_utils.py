"""
@Project :acouinput_python
@File ：common_utils.py
@Date ： 2022/4/21 15:44
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import numpy as np
from scipy.signal import savgol_filter

from utils.plot_utils import show_d_cir, show_signals, show_segmentation
from constants.constants import TAP_SIZE, LABEL_CLASSES, DataType, STEP, SignalPaddingType, WINDOW_SIZE
import torch
import matplotlib.pyplot as plt


def smooth_data(data, win_length=7, poly_order=2):
    data = savgol_filter(data, window_length=win_length, polyorder=poly_order, mode="nearest")
    return data


def segmentation(data):
    shape = np.shape(data)  # (120, N) or (60, N)
    tap_size = shape[0]
    frame_num = shape[1]
    window_num = frame_num // STEP
    alpha = 0.4
    log_ste_frame = 10 * np.log(np.sum(np.square(data), axis=0))  # the log-STE of each frame
    # print(log_ste_frame.shape)
    log_ste_window = np.reshape(log_ste_frame[:window_num*STEP], (window_num, STEP))
    log_ste_window = np.sum(log_ste_window, axis=1) / STEP  # the avg log-STE of each window
    thr = np.zeros(window_num)
    # show_d_cir(data)
    thr[0] = log_ste_window[0]
    for i in range(1, window_num):  # get the dynamic threshold of sliding window
        thr[i] = (1 - alpha) * thr[i-1] + alpha * log_ste_window[i]
    log_ste_thr = np.reshape(np.repeat(thr, STEP), -1, order='F')
    log_ste_thr = np.concatenate((log_ste_thr, np.ones(frame_num - window_num*STEP) * log_ste_thr[-1]), axis=0)
    is_higher_than_thr = log_ste_frame >= log_ste_thr
    # show_segmentation(log_ste_frame, log_ste_thr)
    # show_signals(is_higher_than_thr)
    letter_frame_num = 40  # 字符的最小长度
    letter_gap_frame_num = 35  # 字符间的间隔最小长度
    word_gap_frame_num = 80  # 单词的间隔最小长度
    # 排除掉非常短的字符片段
    # 1. 找到第一个非0的值 -> 第一个字符的起点
    index = 0
    while index < frame_num and not is_higher_than_thr[index]:
        index = index + 1
    letter_start_index = index  # 字符的起始索引（第一个高于阈值的采样点索引）
    res = []
    # 2. 遍历所有的帧，当is_higher_than_thr时跳过，并更新end_index，
    # 找到所有的间隔
    while index < frame_num:
        # 1) 找到下一个低于阈值的采样点，即为gap的起始点
        while index < frame_num and is_higher_than_thr[index]:
            index = index + 1
        gap_start_index = index
        # 2) 一直遍历gap，找到第一个高于阈值的采样点
        # 如果gap的长度高于阈值，则视为一个字符，否则当前间隔视为字符的一部分
        while index < frame_num and not is_higher_than_thr[index]:
            index = index + 1
        gap_end_index = index
        curr_gap_len = gap_end_index - gap_start_index
        if curr_gap_len >= letter_gap_frame_num or index == frame_num:  # 当前字符结束 且当前单词结束
            curr_letter_len = gap_start_index - letter_start_index  # 不包含gap_start_index
            if curr_letter_len >= letter_frame_num:
                letter_end_index = gap_start_index
                res.append([letter_start_index, letter_end_index])
                letter_start_index = gap_end_index
                if curr_gap_len >= word_gap_frame_num:
                    # print("出现单词")
                    pass  # todo 处理当前单词 进行拼写纠错
    return res


def is_static(one_step_data, threshold):  # 每次传入(20, 120)的数据
    one_step_data_std = smooth_data(np.var(one_step_data, axis=0))
    # one_step_data_energy = smooth_data(np.sum(one_step_data, axis=0))
    # one_step_data_energy = smooth_data(np.sum(one_step_data, axis=0))
    show_signals(one_step_data_std)
    # show_signals(one_step_data_energy)
    show_signals(one_step_data_std > threshold)
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
    # print(decode_predictions(torch.tensor([[0, 0, 2, 3, 3, 3]]).cuda()))
    pass
# 30: {'c': 38, 'e': 14, 'f': 9, 'g': 2, 'j': 1, 'k': 1, 'l': 24, 's': 24, 'u': 6}
# 40: {'c': 76, 'e': 20, 'f': 4, 'g': 2, 'l': 51, 's': 38, 'u': 5, 'v': 6, 'y': 1}
