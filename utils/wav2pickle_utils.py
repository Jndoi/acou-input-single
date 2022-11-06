"""
@Project :acouinput_python
@File ：wav2csv_pickle_utils.py
@Date ： 2022/4/18 23:16
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from constants.constants import LabelVocabulary, START_INDEX_SHIFT
import os
from tqdm import tqdm
import numpy as np
from utils.plot_utils import show_d_cir, show_signals
from transceiver.receiver import Receiver
import pickle


class DataItem:
    def __init__(self, label, split_abs_d_cir):
        self.label = label  # the index arr of label just like [1, 2] (start with 1)
        self.split_abs_d_cir = split_abs_d_cir


class DataItemWithPhase(DataItem):  # contains real phase
    def __init__(self, label, split_abs_d_cir, split_real_phase):
        super().__init__(label, split_abs_d_cir)
        self.split_real_phase = split_real_phase


def wav2pickle_new(wav_base_path, dump_path=None,
                    start_index_shift=START_INDEX_SHIFT, augmentation_radio=None):
    data = []
    for root, dirs, files in os.walk(wav_base_path):
        for file in tqdm(files, desc=root):
            if os.path.splitext(file)[1] == '.wav':
                label = file.split("_")[0]
                label_0 = ord(label[0]) - ord('a')
                label_1 = ord(label[1]) - ord('a')
                split_abs_d_cir = Receiver.receive_real_time(root, file,
                                                             start_index_shift=start_index_shift,
                                                             augmentation_radio=augmentation_radio)
                data.append(DataItem(label_0, split_abs_d_cir[0]))
                data.append(DataItem(label_1, split_abs_d_cir[1]))
    if dump_path:
        if dump_path:
            pickle.dump(data, open(dump_path, 'wb'))


def wav2pickle(wav_base_path, seq=False, dump_path=None,
               start_index_shift=START_INDEX_SHIFT, augmentation_radio=None):
    data = []
    for root, dirs, files in os.walk(wav_base_path):
        for file in tqdm(files, desc=root):
            if os.path.splitext(file)[1] == '.wav':
                label = file.split("_")[0]
                if seq:
                    label_int = []
                    for label_item in label:
                        label_int.append(ord(label_item) - ord('a') + 1)
                else:
                    label_int = ord(label[0]) - ord('a')
                split_abs_d_cir = Receiver.receive(root, file, gen_img=False,
                                                   start_index_shift=start_index_shift,
                                                   augmentation_radio=augmentation_radio)
                if dump_path:
                    data.append(DataItem(label_int, split_abs_d_cir))
    if dump_path:
        if dump_path:
            pickle.dump(data, open(dump_path, 'wb'))


def wav2pickle_with_phase(wav_base_path, seq=False, dump_path=None,
                          start_index_shift=START_INDEX_SHIFT, augmentation_radio=None):
    data = []
    for root, dirs, files in os.walk(wav_base_path):
        for file in tqdm(files, desc=root):
            if os.path.splitext(file)[1] == '.wav':
                label = file.split("_")[0]
                if seq:
                    label_int = []
                    for label_item in label:
                        label_int.append(ord(label_item) - ord('a') + 1)
                else:
                    label_int = ord(label[0]) - ord('a')
                split_abs_d_cir, split_real_phase = Receiver.\
                    receive_with_real_phase(root, file, gen_img=False,
                                            start_index_shift=start_index_shift,
                                            augmentation_radio=augmentation_radio)
                if dump_path:
                    data.append(DataItemWithPhase(label_int, split_abs_d_cir, split_real_phase))
    if dump_path:
        if dump_path:
            pickle.dump(data, open(dump_path, 'wb'))


def wav2pickle_with_angle(wav_base_path, seq=False, dump_path=None,
                          start_index_shift=START_INDEX_SHIFT, augmentation_radio=None):
    data = []
    for root, dirs, files in os.walk(wav_base_path):
        for file in tqdm(files, desc=root):
            if os.path.splitext(file)[1] == '.wav':
                label = file.split("_")[0]
                if seq:
                    label_int = []
                    for label_item in label:
                        label_int.append(ord(label_item) - ord('a') + 1)
                else:
                    label_int = ord(label[0]) - ord('a')
                split_abs_d_cir, split_real_phase = Receiver.\
                    receive_with_real_phase(root, file, gen_img=False,
                                            start_index_shift=start_index_shift,
                                            augmentation_radio=augmentation_radio)
                if dump_path:
                    data.append(DataItemWithPhase(label_int, split_abs_d_cir, split_real_phase))
    if dump_path:
        if dump_path:
            pickle.dump(data, open(dump_path, 'wb'))


def load_data_from_pickle(base_path=None):
    if base_path:
        return pickle.load(open(base_path, 'rb'))
    else:
        raise Exception("Base path is None")


if __name__ == '__main__':
    # pass
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single_",
    #            dump_path=r'../data/dataset_single_smooth_20_40.pkl',
    #            start_index_shift=START_INDEX_SHIFT)
    # wav2pickle_new(wav_base_path=r"D:\AcouInputDataSet\dataset",
    #                dump_path=r'../data/dataset.pkl', start_index_shift=START_INDEX_SHIFT)
    wav2pickle_new(wav_base_path=r"D:\AcouInputDataSet\dataset",
                   dump_path=r'../data/dataset_10cm.pkl', start_index_shift=START_INDEX_SHIFT+14)
    wav2pickle_new(wav_base_path=r"D:\AcouInputDataSet\dataset",
                   dump_path=r'../data/dataset_20cm.pkl', start_index_shift=START_INDEX_SHIFT+28)
    wav2pickle_new(wav_base_path=r"D:\AcouInputDataSet\dataset",
                   dump_path=r'../data/dataset_four_fifth.pkl', start_index_shift=START_INDEX_SHIFT,
                   augmentation_radio=0.8)
    wav2pickle_new(wav_base_path=r"D:\AcouInputDataSet\dataset",
                   dump_path=r'../data/dataset_five_fourth.pkl', start_index_shift=START_INDEX_SHIFT,
                   augmentation_radio=1.25)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_20cm.pkl', start_index_shift=START_INDEX_SHIFT + 28)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_10cm.pkl', start_index_shift=START_INDEX_SHIFT + 14)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_four_fifth.pkl', start_index_shift=START_INDEX_SHIFT,
    #            augmentation_radio=0.8)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_five_fourth.pkl', start_index_shift=START_INDEX_SHIFT,
    #            augmentation_radio=1.25)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_20cm_four_fifth.pkl',
    #            start_index_shift=START_INDEX_SHIFT + 28,
    #            augmentation_radio=0.8)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_10cm_four_fifth.pkl',
    #            start_index_shift=START_INDEX_SHIFT + 14,
    #            augmentation_radio=0.8)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_20cm_five_fourth.pkl',
    #            start_index_shift=START_INDEX_SHIFT + 28,
    #            augmentation_radio=1.25)
    # wav2pickle(wav_base_path=r"D:\AcouInputDataSet\single",
    #            dump_path=r'../data/dataset_single_smooth_20_40_10cm_five_fourth.pkl',
    #            start_index_shift=START_INDEX_SHIFT + 14,
    #            augmentation_radio=1.25)
