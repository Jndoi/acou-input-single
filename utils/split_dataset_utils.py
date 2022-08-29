"""
@Project :acouinput_python
@File ：split_dataset.py
@Date ： 2022/6/18 15:56
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import os
import shutil
from sklearn.model_selection import train_test_split
from transceiver.receiver import Receiver
from tqdm import tqdm


class Flag(object):
    SplitDataSet = "SplitDataSet"
    GenPhaseImg = "GenPhaseImg"
    GenDCIRImg = "GenDCIRImg"


if __name__ == '__main__':
    flag = Flag.GenDCIRImg
    data_path = r"D:\AcouInputDataSet\single"
    d_cir_img_root_path = r"D:\AcouInputDataSet\single_img_energy"
    phase_img_root_path = r"D:\AcouInputDataSet\all_phase_img"
    train_root_path = r"D:\AcouInputDataSet\train"
    test_root_path = r"D:\AcouInputDataSet\test"
    if flag == Flag.SplitDataSet:
        for root, dirs, files in os.walk(data_path):
            if files:
                current_label = root.split("\\")[-1]
                current_train_folder = os.path.join(train_root_path, current_label)
                current_test_folder = os.path.join(test_root_path, current_label)
                if not os.path.exists(current_train_folder):
                    os.makedirs(current_train_folder)
                if not os.path.exists(current_test_folder):
                    os.makedirs(current_test_folder)
                x_train, x_test = train_test_split(files, train_size=0.7, random_state=0)
                for file in x_train:
                    shutil.copy(os.path.join(root, file), os.path.join(current_train_folder, file))
                for file in x_test:
                    shutil.copy(os.path.join(root, file), os.path.join(current_test_folder, file))
    elif flag == Flag.GenDCIRImg:
        for root, dirs, files in os.walk(data_path):
            if files:
                current_label = root.split("\\")[-1]
                img_folder = os.path.join(d_cir_img_root_path, current_label)
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                for file in tqdm(files, desc=root):
                    Receiver.receive(root, file, gen_img=True, img_save_path=img_folder)
    elif flag == Flag.GenPhaseImg:
        for root, dirs, files in os.walk(data_path):
            if files:
                current_label = root.split("\\")[-1]
                img_folder = os.path.join(phase_img_root_path, current_label)
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                for file in tqdm(files, desc=root):
                    Receiver.receive(root, file, gen_img=False, gen_phase=True, img_save_path=img_folder)