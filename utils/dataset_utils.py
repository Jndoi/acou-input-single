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
from utils.wav2pickle_utils import load_data_from_pickle, DataItem
from utils.plot_utils import show_d_cir
from utils.common_utils import padding_batch_signals


# set the random seed to ensure the data is same
torch.manual_seed(0)
TRAIN_RATE = 0.8
VAL_RATE = 0.1
TEST_RATE = 0.1


def get_data_from_pickles(data_path=None, data_type=DEFAULT_CONFIG.get("DataType")):
    data_d_cir = []
    # data_real_phase = []  # used for phase
    data_label = []
    for data_path_item in data_path:
        if data_type == DataType.AbsDCirAndRealPhase:
            data_d_cir_item, data_label_item = get_data_from_pickle(data_path_item, data_type)
            data_d_cir.extend(data_d_cir_item)
            data_label.extend(data_label_item)
        else:
            raise Exception("data type error")
    return data_d_cir, data_label


def get_data_from_pickle(data_path=None, data_type=DEFAULT_CONFIG.get("DataType")):
    data_d_cir = []
    # data_real_phase = []
    data_label = []
    load_data = load_data_from_pickle(data_path)
    if data_type == DataType.AbsDCirAndRealPhase:
        for item in load_data:
            data_label.append(item.label)
            data_d_cir.append(item.split_abs_d_cir)
            # data_real_phase.append(item.split_real_phase)
    else:
        raise Exception("Data Type must be AbsDCirAndRealPhase, please swap it in DataItem")
    return data_d_cir, data_label


class AcouInputDataset(Dataset):
    def __init__(self, data_path=None, transform=None):
        if not isinstance(data_path, type([])):
            data_path = [data_path]
        self.d_cir_data, self.label = get_data_from_pickles(data_path, DataType.AbsDCirAndRealPhase)
        self.transform = transform

    def __getitem__(self, index):
        d_cir_item = self.d_cir_data[index]
        # phase_item = self.phase_data[index]
        item_label = self.label[index]  # get the label
        if self.transform is not None:
            d_cir_item = self.transform(d_cir_item)
            # phase_item = self.transform(phase_item)
        # return d_cir_item, phase_item, item_label
        return d_cir_item, item_label

    def __len__(self):
        return len(self.label)  # size of dataset


def data_frames_collate_fn(data):
    # data is a list, and each item in data is a tuple (d_cir_x, phase_x, y)
    # x stands for the features, and y stands for the labels
    return padding_batch_signals(data)


def get_data_loader(loader_type=DatasetLoadType.ALL,
                    batch_size=1,
                    data_path=None,
                    data_path_train=None,
                    data_path_test=None,
                    drop_last=True):
    collate_fn = data_frames_collate_fn
    if loader_type == DatasetLoadType.ALL:
        data_set = AcouInputDataset(data_path)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, drop_last=drop_last)
        return data_loader
    elif loader_type == DatasetLoadType.TrainAndTest:
        if data_path_train and data_path_test:
            train_dataset = AcouInputDataset(data_path_train)
            test_dataset = AcouInputDataset(data_path_test)
        else:
            data_set = AcouInputDataset(data_path)
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
        data_set = AcouInputDataset(data_path)
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
    for index, (train_d_cir_x, train_y) in enumerate(train_data_loader):
        print(train_d_cir_x.shape)
        show_d_cir(train_d_cir_x[0], is_frames=True)
        print(train_y.shape)
        # train_x = train_x.reshape(5, -1, 61, 40)
        break
