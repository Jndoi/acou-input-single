"""
@Project :acouinput_code
@File ：receiver.py
@Date ： 2022/3/30 23:55
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from utils.audio_utils import AudioUtils
from transceiver.transmitter import Transmitter
from constants.constants import *
from utils.common_utils import segmentation
from utils.plot_utils import show_signals, show_d_cir, show_fft, show_phase, save_d_cir
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
from tqdm import tqdm
from utils.data_augmentation_utils import augmentation_speed


class Receiver(object):
    @classmethod
    def gen_d_cir_img(cls, d_cir, base_path="", file_name=None, is_frames=False):
        """
        generate dCIR image
        :param is_frames:
        :param d_cir: difference CIR
        :param base_path: the base path want to save
        :param file_name: the save path
        :return:
        """
        # plt.pcolormesh(np.abs(d_cir), cmap='jet')
        # plt.colorbar()
        # plt.show()
        # plt.figure(figsize=(1, 1), dpi=64)
        if is_frames:
            d_cir = d_cir.squeeze()
            d_cir = np.transpose(d_cir, [1, 0, 2])
            d_cir = np.reshape(d_cir, (d_cir.shape[0], -1), order='C')
        plt.axis('off')
        plt.pcolormesh(np.abs(d_cir), cmap='jet')  # gray
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if not file_name:
            file_name = "default.jpg"
        # print(os.path.join(base_path, file_name))
        plt.savefig(os.path.join(base_path, file_name))  # save the image
        plt.clf()  # clear the canvas
        plt.close()

    @classmethod
    def gen_phase_img(cls, phase, base_path="", file_name=None):
        """
        generate dCIR image
        :param phase: difference CIR
        :param base_path: the base path want to save
        :param file_name: the save path
        :return:
        """
        # plt.pcolormesh(np.abs(d_cir), cmap='jet')
        # plt.colorbar()
        # plt.show()
        # plt.figure(figsize=(1, 1), dpi=64)
        plt.axis('off')
        plt.plot(phase)  # gray
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if not file_name:
            file_name = "default.jpg"
        # print(os.path.join(base_path, file_name))
        plt.savefig(os.path.join(base_path, file_name))  # save the image
        plt.clf()  # clear the canvas
        plt.close()

    @classmethod
    def select_dynamic_tap_by_ste(cls, real_phase, abs_d_cir, win_size=20):
        """
        select select dynamic tap by energy and returns the selected phase
        :param win_size: the window size of short time energy to select the dynamic tap
        :param real_phase: the real phase of bottom
        :param abs_d_cir: the dCIR
        :return: the one-dim selected real phase
        """
        tap_size = abs_d_cir.shape[0]
        slice_num = int(np.floor(real_phase.shape[1] / win_size))
        h_ = abs_d_cir[:, 0: win_size * slice_num]  # 0 to slice_len * slice_num -1
        # calculate the ste of each  taps
        h_ = h_.reshape((tap_size, win_size, -1), order='F')
        dynamic_tap = np.argmax(np.sum(h_, axis=1), axis=0)
        # assign the maximum tap value to each element of slice
        # np.tile: repeating dynamic_tap arr in (win_size, 1) times
        dynamic_tap = np.tile(dynamic_tap, (win_size, 1)).flatten(order='F')
        # get the max tap of last slice and append the dynamic_tap to make the length same as real_phase
        dynamic_tap_last = dynamic_tap[-1]
        phase_len = real_phase.shape[1]
        slices_len = dynamic_tap.shape[0]
        # show_signals(dynamic_tap)
        dynamic_tap = np.r_[dynamic_tap, dynamic_tap_last * np.ones((phase_len - slices_len))].astype(np.int)
        diff_phase = np.diff(real_phase)
        dynamic_phase = np.zeros(phase_len)
        current_phase = 0
        for i in np.arange(1, phase_len):
            current_phase = current_phase + diff_phase[dynamic_tap[i], i - 1]
            dynamic_phase[i] = current_phase
        return dynamic_phase

    @classmethod
    def cal_phase(cls, d_cir, phase_thr=DefaultPhaseThreshold):
        """
        calculate the absolute phase and relative phase from dCIR
        :param d_cir: np.array, complex 2D, the dCIR of wireless channel
        :param phase_thr: double, the threshold of phase difference, exceeding the threshold represents
        a phase change of <b>2pi</b>
        :return: the absolute phase and relative phase
        """
        # 1. calculate the negative because of the angle is related to -d(t)/c
        phase = -np.angle(d_cir)  # the shape of h is (tap_size, frame_num)
        # show_signals(-np.angle(d_cir[0, :]))
        tap_size, frame_num = np.shape(d_cir)
        # make the initial phase begins at 0
        phase = phase - np.tile(phase[:, 0].reshape(tap_size, 1), frame_num)  # phase[:, 0] extend along the X axis
        diff_phase = np.diff(phase)
        real_phase = np.zeros((tap_size, frame_num))
        current_period = np.zeros(tap_size)
        # 2. If the phase difference exceeds the threshold, it will add or subtract 2pi
        # the default value of phase_thr is 0.6 since finger move speed less than 0.25 m/s
        for i in np.arange(0, tap_size):
            for j in np.arange(1, frame_num):  # the real phase at index 0 is 0, so the index begins at 1
                if diff_phase[i, j - 1] > phase_thr * np.pi:
                    current_period[i] = current_period[i] - 2
                elif diff_phase[i, j - 1] < -phase_thr * np.pi:
                    current_period[i] = current_period[i] + 2
                else:
                    pass
                real_phase[i, j] = phase[i, j] + current_period[i] * np.pi
        return real_phase

    @classmethod
    def demodulation(cls, data):
        """
        IQ demodulation
        :param data: the data to be demodulation
        :return: complex demodulated data
        """
        sin_ = AudioUtils.build_cos_or_sin(signal_length=data.shape[0], sin_or_cos=SinOrCosType.Sin)
        cos_ = AudioUtils.build_cos_or_sin(signal_length=data.shape[0], sin_or_cos=SinOrCosType.Cos)
        pass_in_phase = data * cos_
        pass_quadrature = -1 * data * sin_
        base_in_phase = AudioUtils.low_pass(pass_in_phase)
        base_quadrature = AudioUtils.low_pass(pass_quadrature)
        # generate the complex base signals
        return base_in_phase + 1j * base_quadrature

    @classmethod
    def gen_training_matrix(cls, p_value=P_VALUE, l_value=L_VALUE, signal_type=SignalType.Barker):
        """
        generate training matrix according to signal type (SignalType.Barker or SignalType.GSM)
        :param p_value: reference length, default 192
        :param l_value: guard period, default 120 (the number of taps)
        :param signal_type:
        :return:
        """
        training_matrix = np.zeros(p_value * l_value)
        training_seq = Transmitter.get_baseband_sequence(signal_type)
        for i in np.arange(0, p_value):
            training_matrix[i * l_value:i * l_value + l_value] = np.flip(training_seq[i:i + l_value])
        training_matrix = training_matrix.reshape((l_value, p_value), order='F').T
        training_matrix_ = training_matrix.T  # 1/P*M'
        # ((M')*M)\(M') doesn't work
        # training_matrix_ = np.dot(np.linalg.pinv(np.dot(training_matrix.T, training_matrix)), training_matrix.T)
        return training_matrix_

    @classmethod
    def cal_d_cir(cls, complex_y, training_matrix_, frame_len=FRAME_LENGTH, p_value=P_VALUE, l_value=L_VALUE):
        """
        calculate the cir by h = (1/P) M' * yL
        :param training_matrix_: the training matrix
        :param complex_y: the complex signal
        :param frame_len: the frame len
        :param p_value: P, reference length, default 192
        :param l_value: L,  guard period (the number of taps), default 120
        :return:
        """
        frame_num = int(np.floor(complex_y.shape[0] / frame_len))
        complex_y = complex_y[:frame_len * frame_num]
        complex_y_ = complex_y.reshape((frame_len, frame_num), order='F')
        complex_y_ = complex_y_[l_value: l_value + p_value, :]
        # h = np.dot(training_matrix_, complex_y_)
        h = 1 / p_value * np.dot(training_matrix_, complex_y_)
        return np.diff(h)

    @classmethod
    def smooth_data(cls, data, win_length=5, poly_order=3):
        """
        apply a Savitzky-Golay filter to smooth the data
        :param data: data to be filtered
        :param win_length: the window size of filter,
        `window_length` must be a positive odd integer. If `mode` is 'interp'
        :param poly_order: the order of the polynomial used to fit the samples.
        :return: the filtered data
        """
        # print(data.shape)
        # print(data.T.shape)
        # show_signals(data)
        # data = cv2.GaussianBlur(src=data, ksize=(5, 5), sigmaX=0, sigmaY=0.5)
        data = savgol_filter(data, window_length=win_length, polyorder=poly_order, mode="nearest")
        # show_signals(data)
        return data

    @classmethod
    def remove_static_d_cir(cls, d_cir, thr=0.02):
        """
        remove the static part of signal according to the sum ste of each taps by threshold-based method
        :param d_cir: the d cir
        :param thr: the threshold of sum ste
        :return: removed static d cir
        """
        frame_num = d_cir.shape[1]
        sum_abs_d_cir = cls.smooth_data(sum(abs(d_cir)))
        # show_signals(sum_abs_d_cir)
        motion_frames = sum_abs_d_cir > thr
        new_d_cir = d_cir * motion_frames
        for i in np.arange(1, frame_num):  # set the zeros diff_h to the forward diff_h value
            if sum(new_d_cir[:, i]) == 0:
                new_d_cir[:, i] = new_d_cir[:, i - 1]
        return new_d_cir

    @classmethod
    def get_signals_by_filename(cls, base_path, filename, signal_type=SignalType.Barker,
                                start_index_shift=START_INDEX_SHIFT, device_type=DeviceType.HONOR30Pro):
        """
        get the signals from the wav file
        :param device_type:
        :param start_index_shift:
        :param signal_type: SignalType.Barker or SignalType.GSM
        :param base_path: the base path of wav file
        :param filename: the filename of wav file
        :return: the filtered and aligned signals
        """
        # 1. get data from bottom mic
        if filename is None or len(filename) == 0:
            file_path = base_path
        else:
            file_path = os.path.join(base_path, filename)
        # data[:, 0] received from top mic, data[:, 1] received from bottom mic [phone: HONOR 30 pro]
        data = AudioUtils.read_audio(file_path, device_type)
        # 2. remove the audible noise
        filtered_data = AudioUtils.band_pass(data)
        # 3. align signals
        send_signal = AudioUtils.band_pass(Transmitter.get_passband_sequence(signal_type))
        start_index = AudioUtils.align_signal(filtered_data, send_signal, 0.1)  # get the arrived time on direct path
        # print(start_index)
        # start_index = 0
        if start_index >= start_index_shift:  # remove the path length from speaker to bottom mic
            start_index = start_index - start_index_shift
        return filtered_data[start_index:]

    @classmethod
    def padding_signals(cls, data, data_type, window_size=WINDOW_SIZE, step=STEP):
        """
        padding signals by zero-padding (dcir) or last-data padding (phase)
        """
        if data is None:
            raise Exception("fail to pad signals: provided data is None")

        padding_method = SignalPaddingTypeMap.get(data_type)
        if padding_method == SignalPaddingType.LastValuePadding:
            if len(data.shape) == 2:
                padding_len = (step - data.shape[1] % step) if data.shape[1] > window_size \
                    else window_size - data.shape[1] % step
                padding_value = data[:, -1]
                return np.c_[data, np.tile(padding_value, (1, padding_len))]
            elif len(data.shape) == 1:
                padding_len = (step - data.shape[0] % step) if data.shape[0] > window_size \
                    else window_size - data.shape[0] % step
                padding_value = np.ones(1) * data[-1]
                return np.r_[data, np.tile(padding_value, padding_len)]
            else:
                raise Exception("fail to pad signals: the shape {} of data is valid".format(data.shape))
        elif padding_method == SignalPaddingType.ZeroPadding:
            if len(data.shape) == 2:
                padding_len = (step - data.shape[1] % step) if data.shape[1] > window_size \
                    else window_size - data.shape[0] % step
                padding_value = np.zeros((data.shape[0], 1))
                return np.c_[data, np.tile(padding_value, (1, padding_len))]
            elif len(data.shape) == 1:
                padding_len = (step - data.shape[0] % step) if data.shape[0] > window_size \
                    else window_size - data.shape[0] % step
                padding_value = np.zeros(1)
                return np.r_[data, np.tile(padding_value, padding_len)]
            else:
                raise Exception("fail to pad signals: the shape {} of data is valid".format(data.shape))
        else:
            raise Exception("fail to pad signals: padding method {} must be one of {}".
                            format(padding_method, [SignalPaddingType.LastValuePadding, SignalPaddingType.ZeroPadding]))

    @classmethod
    def normalization(cls, data):
        """
        min-max normalization
        :param data:
        :return:
        """
        _range = np.max(data) - np.min(data)
        if _range == 0.0:
            return data
        return data / _range

    @classmethod
    def split_abs_d_cir(cls, abs_d_cir, window_size=WINDOW_SIZE, step=STEP):
        # abs_d_cir_ = abs_d_cir.reshape((2, -1), order="F")
        # abs_d_cir_ = np.mean(abs_d_cir_, axis=0)
        # abs_d_cir = abs_d_cir_.reshape((abs_d_cir.shape[0] // 2, -1), order="F")
        abs_d_cir_h = abs_d_cir.shape[0]
        abs_d_cir = cls.padding_signals(abs_d_cir, DataType.AbsDCir, window_size, step)
        seq_len = int(np.floor((abs_d_cir.shape[1] - window_size) / step)) + 1
        res = np.zeros((seq_len, 1, abs_d_cir_h, window_size))  # split signal by window
        for i in np.arange(0, seq_len):
            res[i, 0, :, :] = cls.normalization(abs_d_cir[:, i * step: i * step + window_size]) * 255
        return res.astype(np.uint8)

    @classmethod
    def split_abs_d_cir_phase(cls, abs_d_cir, real_phase, window_size=WINDOW_SIZE, step=STEP):
        """

        :param abs_d_cir: the mode of difference cir
        :param real_phase: real phase
        :param window_size: window size, default=600ms
        :param step: move step, default=200ms
        :return: (N, 120, window_size) (N, 1, window_size) where
        """
        # max pooling (tap_size, window_size) -> (tap_size//2, window_size)
        abs_d_cir_ = abs_d_cir.reshape((2, -1), order="F")
        abs_d_cir_ = np.max(abs_d_cir_, axis=0)
        abs_d_cir = abs_d_cir_.reshape((abs_d_cir.shape[0] // 2, -1), order="F")
        abs_d_cir_h = abs_d_cir.shape[0]
        abs_d_cir = cls.padding_signals(abs_d_cir, DataType.AbsDCir, window_size, step)
        real_phase = cls.padding_signals(real_phase, DataType.RealPhase, window_size, step)
        # 1 stands for 0.025cm, the value range is -0.818m~0.818m
        real_phase = (real_phase * WINDOW_SIZE).astype(np.int16)
        seq_len = int(np.floor((real_phase.shape[0] - window_size) / step)) + 1
        d_cir = np.zeros((seq_len, 1, abs_d_cir_h, window_size))  # split signal by window
        phase = np.zeros((seq_len, window_size))  # split signal by window
        for i in np.arange(0, seq_len):
            phase[i, :] = real_phase[i * step: i * step + window_size]
            d_cir[i, 0, :, :] = cls.normalization(abs_d_cir[:, i * step: i * step + window_size])*255
        # split_abs_d_cir.squeeze(-3) add channel dim
        return d_cir.astype(np.uint8), phase.astype(np.int16)

    @classmethod
    def down_sampling(cls, data):
        data = data.reshape((2, -1), order="F")
        data = np.max(data, axis=0)
        data = data.reshape((60, -1), order="F")
        return data

    @classmethod
    def receive(cls, base_path, filename, gen_img=False, img_save_path=None,
                start_index_shift=START_INDEX_SHIFT, augmentation_radio=None, device_type=DeviceType.HONOR30Pro):
        data = cls.get_signals_by_filename(base_path, filename, start_index_shift=start_index_shift,
                                           device_type=device_type)
        data = cls.cal_d_cir(cls.demodulation(data), cls.gen_training_matrix())
        data = cls.smooth_data(np.real(data)) + 1j * cls.smooth_data(np.imag(data))
        data = cls.down_sampling(data)
        data_abs = np.abs(data)
        # show_d_cir(data_abs)
        segmentation_index = segmentation(data_abs)
        letter_num = len(segmentation_index)
        if letter_num != 1:
            return 1
        else:
            return 0
        # segmentation_data = []
        # for index in range(letter_num):
        #     curr_data_abs = data_abs[:, segmentation_index[index][0]:segmentation_index[index][1]]
        #     if augmentation_radio:
        #         curr_data_abs = augmentation_speed(curr_data_abs, speed_radio=augmentation_radio)
        #     curr_data_abs = cls.split_abs_d_cir(curr_data_abs)
        #     segmentation_data.append(curr_data_abs)
        # return segmentation_data

    @classmethod
    def receive_real_time(cls, base_path, filename, start_index_shift=START_INDEX_SHIFT, augmentation_radio=None):
        data = cls.get_signals_by_filename(base_path, filename, start_index_shift=start_index_shift)
        data = cls.cal_d_cir(cls.demodulation(data), cls.gen_training_matrix())
        data = cls.smooth_data(np.real(data)) + 1j * cls.smooth_data(np.imag(data))
        data = cls.down_sampling(data)
        data_abs = np.abs(data)
        # save_d_cir(data_abs)
        # 对信号进行分段[[begin1, end1], [begin2, end2], ..., ]
        segmentation_index = segmentation(data_abs)
        segmentation_data = []
        for index in segmentation_index:
            curr_data_abs = data_abs[:, index[0]:index[1]]
            if augmentation_radio:
                curr_data_abs = augmentation_speed(curr_data_abs, speed_radio=augmentation_radio)
            curr_data_abs = cls.split_abs_d_cir(curr_data_abs)
            segmentation_data.append(curr_data_abs)
        if len(segmentation_data) == 0:
            if augmentation_radio:
                data_abs = augmentation_speed(data_abs, speed_radio=augmentation_radio)
            curr_data_abs = cls.split_abs_d_cir(data_abs)
            segmentation_data.append(curr_data_abs)
        return segmentation_data

    @classmethod
    def receive_with_real_phase(cls, base_path, filename, gen_img=False, img_save_path=None,
                                start_index_shift=START_INDEX_SHIFT, augmentation_radio=None,
                                device_type=DeviceType.HONOR30Pro):
        data = cls.get_signals_by_filename(base_path, filename, start_index_shift=start_index_shift,
                                           device_type=device_type)
        data = cls.cal_d_cir(cls.demodulation(data), cls.gen_training_matrix())
        data = cls.smooth_data(np.real(data)) + 1j * cls.smooth_data(np.imag(data))
        abs_d_cir = np.abs(data)
        real_phase = cls.cal_phase(data)
        real_phase = cls.select_dynamic_tap_by_ste(real_phase, abs_d_cir, STEP // 2) * WaveLength * 25 / np.pi  # unit cm
        real_phase = cls.smooth_data(real_phase)
        if augmentation_radio:
            abs_d_cir = augmentation_speed(abs_d_cir, speed_radio=augmentation_radio)
            real_phase = augmentation_speed(real_phase, speed_radio=augmentation_radio)
        split_abs_d_cir, split_real_phase = cls.split_abs_d_cir_phase(abs_d_cir, real_phase)
        if gen_img:
            cls.gen_d_cir_img(abs_d_cir, base_path=img_save_path,
                              file_name=filename.split('.')[0] + '.png', is_frames=True)
        return split_abs_d_cir, split_real_phase


def gen_cir(base_path, label_arr):
    for label in label_arr:
        folder_path = os.path.join(base_path, label)
        for root, dirs, files in os.walk(folder_path):
            for file in tqdm(files, desc="generate cir of char '{}'".format(label)):
                if os.path.splitext(file)[1] == '.wav':  # find all wav files
                    Receiver.receive(folder_path, file, label)


if __name__ == '__main__':
    # letter_dict = {}
    # count = 0
    # for root, dirs, files in os.walk(r"D:\AcouInputDataSet\tmp2"):
    #     for file in tqdm(files):
    #         if os.path.splitext(file)[1] == '.wav':
    #             label = file.split("_")[0]
    #             val = (Receiver.receive(
    #                 root, file, gen_img=False,
    #                 start_index_shift=START_INDEX_SHIFT))
    #             if val == 1:
    #                 os.remove(os.path.join(r"D:\AcouInputDataSet\tmp2", file))
    #                 if label not in letter_dict:
    #                     letter_dict[label] = 0
    #                 letter_dict[label] = letter_dict[label]+1
    #                 count += val
    # print(count)
    # print(letter_dict)

    segmentation_data = Receiver.receive_real_time(
            base_path=r'D:\Program\Tencent\QQ-Chat-Record\563496927\FileRecv\MobileFile',
            filename='test_1671784067670.wav',  # away from and close to 10cm
            # filename='a_1671537161369.wav',  # check the data augmentation
            # filename='a word_1667371289681.wav',  # test the segmentation
            start_index_shift=START_INDEX_SHIFT,
        )  # (30, 1, 60, 40)
    # for data in segmentation_data:
    #     save_d_cir(data, is_frames=True)
