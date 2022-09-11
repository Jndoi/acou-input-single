"""
@Project :acouinput_python
@File ：dataset_utils_new.py
@Date ： 2022/5/27 13:22
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from constants.constants import DatasetLoadType, DataType, DEFAULT_CONFIG
from utils.wav2pickle_utils import load_data_from_pickle, DataItem, DataItemWithPhase
from utils.plot_utils import show_d_cir, show_signals
from utils.common_utils import padding_batch_signals


# set the random seed to ensure the data is same
torch.manual_seed(0)
TRAIN_RATE = 0.8
VAL_RATE = 0.1
TEST_RATE = 0.1


def get_data_from_pickles(data_path=None, data_type=DEFAULT_CONFIG.get("DataType")):
    data_d_cir = []
    data_label = []
    if data_type == DataType.AbsDCir:
        for data_path_item in data_path:
            data_d_cir_item, data_label_item = get_data_from_pickle(data_path_item, data_type)
            data_d_cir.extend(data_d_cir_item)
            data_label.extend(data_label_item)
        return data_d_cir, data_label
    elif data_type == DataType.AbsDCirAndRealPhase:
        data_real_phase = []  # used for phase
        for data_path_item in data_path:
            data_d_cir_item, data_real_phase_item, data_label_item = get_data_from_pickle(data_path_item, data_type)
            data_d_cir.extend(data_d_cir_item)
            data_real_phase.extend(data_real_phase_item)
            data_label.extend(data_label_item)
        return data_d_cir, data_real_phase, data_label
    else:
        raise Exception("Data Type error")


def get_data_from_pickle(data_path=None, data_type=DEFAULT_CONFIG.get("DataType")):
    data_d_cir = []
    data_label = []
    load_data = load_data_from_pickle(data_path)
    if data_type == DataType.AbsDCir:
        for item in load_data:
            data_label.append(item.label)
            data_d_cir.append(item.split_abs_d_cir)
        return data_d_cir, data_label
    elif data_type == DataType.AbsDCirAndRealPhase:
        data_real_phase = []
        for item in load_data:
            data_label.append(item.label)
            data_d_cir.append(item.split_abs_d_cir)
            data_real_phase.append(item.split_real_phase)
        return data_d_cir, data_real_phase, data_label
    else:
        raise Exception("Data Type error")


class AcouInputDatasetFactory:
    @classmethod
    def get_data_set(cls, data_path, data_type=DEFAULT_CONFIG.get("DataType")):
        if data_type == DataType.AbsDCir:
            return AcouInputAbsDCirDataset(data_type, data_path)
        elif data_type == DataType.AbsDCirAndRealPhase:
            return AcouInputAbsDCirAndRealPhaseDataset(data_type, data_path)
        else:
            raise Exception("Data Type error")


class AcouInputDataset(Dataset):
    def __init__(self, data_type=DEFAULT_CONFIG.get("DataType"), data_path=None):
        if not isinstance(data_path, type([])):
            data_path = [data_path]
        self.data_type = data_type
        if data_type == DataType.AbsDCir:
            self.d_cir_data, self.label = get_data_from_pickles(data_path, data_type)
        elif data_type == DataType.AbsDCirAndRealPhase:
            self.d_cir_data, self.real_phase_data, self.label = get_data_from_pickles(data_path, data_type)
        else:
            raise Exception("Data Type error")

    def __getitem__(self, index):
        return None

    def __len__(self):
        return len(self.label)  # size of dataset


class AcouInputAbsDCirAndRealPhaseDataset(AcouInputDataset):
    def __getitem__(self, index):
        d_cir_item = self.d_cir_data[index]
        phase_item = self.real_phase_data[index]
        item_label = self.label[index]  # get the label
        return d_cir_item, phase_item, item_label


class AcouInputAbsDCirDataset(AcouInputDataset):
    def __getitem__(self, index):
        d_cir_item = self.d_cir_data[index]
        item_label = self.label[index]  # get the label
        return d_cir_item, item_label


def data_frames_collate_fn_abs_d_cir(data):
    # data is a list, and each item in data is a tuple (d_cir_x, y)
    # x stands for the features, and y stands for the labels
    return padding_batch_signals(data, DataType.AbsDCir)


def data_frames_collate_fn_abs_d_cir_real_phase(data):
    # data is a list, and each item in data is a tuple (d_cir_x, phase_x, y)
    # x stands for the features, and y stands for the labels
    return padding_batch_signals(data, DataType.AbsDCirAndRealPhase)


def get_data_loader(loader_type=DatasetLoadType.ALL,
                    batch_size=1,
                    data_path=None,
                    data_path_train=None,
                    data_path_test=None,
                    drop_last=True,
                    data_type=DEFAULT_CONFIG.get("DataType")):
    if data_type == DataType.AbsDCir:
        collate_fn = data_frames_collate_fn_abs_d_cir
    elif data_type == DataType.AbsDCirAndRealPhase:
        collate_fn = data_frames_collate_fn_abs_d_cir_real_phase
    else:
        raise Exception("Data Type error")
    if loader_type == DatasetLoadType.ALL:
        data_set = AcouInputDatasetFactory.get_data_set(data_path)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, drop_last=drop_last)
        return data_loader
    elif loader_type == DatasetLoadType.TrainAndTest:
        if data_path_train and data_path_test:
            train_dataset = AcouInputDatasetFactory.get_data_set(data_path_train)
            test_dataset = AcouInputDatasetFactory.get_data_set(data_path_test)
        else:
            data_set = AcouInputDatasetFactory.get_data_set(data_path)
            train_size = int(len(data_set) * TRAIN_RATE)
            test_size = len(data_set) - train_size
            # torch.manual_seed(0) to ensure the split data is same every time
            train_dataset, test_dataset = random_split(data_set, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=drop_last)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=drop_last)
        return train_loader, test_loader
    elif loader_type == DatasetLoadType.TrainValidAndTest:
        data_set = AcouInputDatasetFactory.get_data_set(data_path)
        train_size = int(len(data_set) * TRAIN_RATE)
        valid_size = int(len(data_set) * VAL_RATE)
        test_size = len(data_set) - train_size - valid_size
        # torch.manual_seed(0) to ensure the split data is same every time
        train_dataset, valid_dataset, test_dataset = random_split(data_set, [train_size, valid_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=drop_last)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, drop_last=drop_last)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=drop_last)
        return train_loader, valid_loader, test_loader
    else:
        raise Exception("fail to get data loader: type must in {}".
                        format([DatasetLoadType.ALL, DatasetLoadType.TrainAndTest]))


if __name__ == '__main__':
    train_data_loader, test_data_loader = \
        get_data_loader(data_path=r"../data/dataset_single_smooth.pkl",
                        loader_type=DatasetLoadType.TrainAndTest,
                        batch_size=8)
    for index, (train_d_cir_x, train_phase_x, train_y) in enumerate(train_data_loader):
        print(train_d_cir_x.shape)
        print(train_phase_x.shape)
        for item in train_d_cir_x:
            show_d_cir(item, is_frames=True)
        for item in train_phase_x:
            show_signals(item.reshape(-1))
        print(train_y.shape)
        # train_x = train_x.reshape(5, -1, 61, 40)
        break
