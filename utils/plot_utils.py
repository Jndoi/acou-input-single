"""
@Project :acouinput_python
@File ：plot_utils.py
@Date ： 2022/4/7 13:28
@Author ： Qiuyang Zeng
@Software ：PyCharm
ref: https://blog.csdn.net/qq_41645987/article/details/109148274
"""
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constants.constants import LabelVocabulary


def show_confusion_matrix(C):
    """

    :param C: result of confusion_matrix
    :return:
    """
    xtick = LabelVocabulary
    ytick = LabelVocabulary
    plt.figure(figsize=(12, 6))
    C = np.around(C, 2)
    # annot=True 代表 在图上显示 对应的值
    # fmt 属性 代表输出值的格式
    # cbar=False, 不显示 热力棒
    sns.heatmap(C, fmt='g', cmap='Blues', annot=True,
                cbar=False, xticklabels=xtick, yticklabels=ytick)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tick_params(bottom=False, top=False, left=False, right=False)  # remove tick marks
    plt.savefig(r"C:\Users\zengq\Desktop\confusion_matrix.pdf", dpi=600)
    plt.show()


def show_fft(signals, **kwargs):
    """
    show the fft result of signals
    :param signals:
    :return:
    """
    plt.plot(abs(fft(signals, **kwargs)), linewidth=0.5)
    plt.show()


def show_signals(signals, is_frames=False):
    """
    plot origin signals
    :param is_frames:
    :param signals:
    :return:
    """
    plt.figure(figsize=(10, 6), dpi=600)
    if is_frames:
        frame = signals[0]
        for index in np.arange(1, signals.shape[0]):
            frame = np.r_[frame, signals[index]]
    else:
        frame = signals
    # frame = frame.squeeze()
    plt.plot(frame, linewidth=0.5)
    plt.margins(0, 0.1)
    # plt.plot(np.ones(frame.shape)*3, linewidth=0.5)
    plt.show()


def show_segmentation(frames, thr):
    plt.figure(figsize=(10, 6), dpi=600)
    # plt.gca().set (gca, 'LooseInset', get(gca, 'TightInset'))
    plt.plot(np.arange(0, frames.shape[0])/100.0, frames, linewidth=1)
    plt.plot(np.arange(0, frames.shape[0])/100.0, thr, linewidth=1, )
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    plt.xlabel("Time (sec)")
    plt.ylabel("log-STE (dB)")
    plt.margins(0, 0.1)
    plt.show()


def show_phase(signals, is_frames=False):
    """
    plot origin signals
    :param is_frames:
    :param signals:
    :return:
    """
    plt.figure(figsize=(10, 6), dpi=200)
    if is_frames:
        frame = signals[0]
        for index in np.arange(1, signals.shape[0]):
            frame = np.r_[frame, signals[index]]
    else:
        frame = signals
    # frame = frame.squeeze()
    plt.plot(np.arange(0, frame.shape[0])/100.0, frame, linewidth=0.5)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    plt.xlabel("Time (sec)")
    plt.ylabel("Distance (cm)")
    plt.show()


def show_d_cir(d_cir, is_frames=False):
    """
    plot dCIR image
    :param is_frames:
    :param d_cir: difference CIR
    :return:
    """
    # the shape of d_cir is (N, 60, 30), we should change it to (60, N*30)
    plt.figure(figsize=(10, 6), dpi=200)
    if is_frames:
        d_cir = d_cir.squeeze()
        d_cir = np.transpose(d_cir, [1, 0, 2])
        d_cir = np.reshape(d_cir, (d_cir.shape[0], -1), order='C')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    plt.xlabel("Time (sec)")
    plt.ylabel("Tap Index")
    plt.pcolormesh(np.arange(0, d_cir.shape[1])/100.0, np.arange(1, d_cir.shape[0]+1), d_cir, cmap='jet', shading='auto')
    # plt.axis('off')
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.show()


if __name__ == '__main__':
    pass
