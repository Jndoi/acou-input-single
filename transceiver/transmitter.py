"""
@Project :acouinput_python
@File ：transmitter.py
@Date ： 2022/4/7 13:27
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from constants.constants import TrainingSequence, SinOrCosType, SignalType
import numpy as np
from numpy.fft import fftshift, fft, ifft
from scipy.interpolate import interp1d
from utils.normalization_utils import normalization
from utils.plot_utils import show_signals, show_fft
from utils.audio_utils import AudioUtils
import pickle
import os
import sys


class Transmitter(object):

    @classmethod
    def get_passband_sequence(cls, signal_type=SignalType.Barker):
        root_path = "../"
        if os.getcwd().endswith("acou-input-single"):
            root_path = ""
        if signal_type == SignalType.GSM:
            return pickle.load(open(root_path+'data/gsm_passband.pkl', 'rb'))
        elif signal_type == SignalType.Barker:
            return pickle.load(open(root_path+'data/barker_passband.pkl', 'rb'))
        else:
            raise Exception("NoSuchTypeError")

    @classmethod
    def get_baseband_sequence(cls, signal_type=SignalType.Barker):
        root_path = "../"
        if os.getcwd().endswith("acou-input-single"):
            root_path = ""
        if signal_type == SignalType.GSM:
            return pickle.load(open(root_path + 'data/gsm_baseband.pkl', 'rb'))
        elif signal_type == SignalType.Barker:
            return pickle.load(open(root_path + 'data/barker_baseband.pkl', 'rb'))
        else:
            raise Exception("NoSuchTypeError")

    @classmethod
    def gen_sequence(cls, signal_type=SignalType.GSM):
        training_seq = TrainingSequence.get(signal_type)
        training_seq_fft = fftshift(fft(training_seq))
        len_up_sample = len(training_seq) * 12
        up_sample_training_seq_fft = np.zeros(len_up_sample, dtype=complex)
        up_sample_training_seq_fft[143:169] = training_seq_fft
        up_sample_training_seq_fft = fftshift(up_sample_training_seq_fft)
        up_sample_training_seq = np.real(ifft(up_sample_training_seq_fft))
        up_sample_training_seq = np.r_[up_sample_training_seq, np.zeros(12*14)]
        # min-max normalization
        up_sample_training_seq = up_sample_training_seq / np.max(np.abs(up_sample_training_seq))
        show_signals(up_sample_training_seq)
        show_fft(up_sample_training_seq)
        up_sample_training_seq_fc = up_sample_training_seq * AudioUtils.build_cos_or_sin(
            len(up_sample_training_seq), SinOrCosType.Cos)
        training_seq_fc = AudioUtils.band_pass(up_sample_training_seq_fc)
        # dump signal
        pickle.dump(up_sample_training_seq, open(os.path.join("../data", signal_type + '_baseband.pkl'), 'wb'))
        pickle.dump(training_seq_fc, open(os.path.join("../data", signal_type+'_passband.pkl'), 'wb'))
        training_seq_fc = np.tile(training_seq_fc, (1, 1000))  # 10sec
        empty_seq = np.zeros((1, training_seq_fc.shape[1]))
        # the dim of 0 is top speaker and the dim of 1 is bottom speaker
        audio_signal = np.r_[training_seq_fc, empty_seq].T
        audio_folder_path = os.path.join(os.path.abspath('..'), "audio")
        if not os.path.exists(audio_folder_path):
            os.mkdir(audio_folder_path)
        AudioUtils.write_audio(audio_signal, os.path.join(audio_folder_path, signal_type + "_frequency.wav"))


if __name__ == '__main__':
    pass
    # Transmitter.gen_sequence(signal_type=SignalType.Barker)
    # Transmitter.get_passband_sequence()
    # Transmitter.get_baseband_sequence()
