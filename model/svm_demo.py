"""
@Project :acouinput_code
@File ：svm_demo.py
@Date ： 2022/4/11 18:21
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
from sklearn import svm
import sklearn
import joblib
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import shutil
import os
import csv


def img2csv(base_path, label_arr):
    """
    > 将img图像的数据转存到csv文件中，每一行代表一个img图像的数据
    > 比如根目录为 r"C:/Users/xxx/Desktop/img"，里面存放了三个标签文件夹 a,b,c，每个文件夹下存放该标签对应的所有图像（matlab生成的图像）
    > 对应的调用方式为 img2csv(r"C:/Users/xxx/Desktop/img", [a, b, c])
    > 程序执行结束后会生成3个csv文件，分别对应标签a、b、c的图像数据，csv文件的每一行都对应一个图像的数据

    read img data and transfer to csv data
    :param base_path:
    :param label_arr: the array of label
    :return:
    """
    for label in label_arr:
        folder_path = os.path.join(base_path, label)
        # create the csv file
        f = open('{}.csv'.format(os.path.join(folder_path, label)), 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(f)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    img = Image.open(os.path.join(root, file))
                    img = img.convert('L')
                    img_arr = np.array(img, dtype=np.uint8)
                    img_arr = np.reshape(img_arr, -1)  # reshape to 1-D
                    csv_writer.writerow(img_arr)


def read_csv(base_path, label_arr):
    """
    > 读取CSV文件，每次读取一个文件，返回一个dict，<label, data>的形式
    > 比如根目录为 r"C:/Users/xxx/Desktop/img"，里面存放了三个标签文件夹 a,b,c，每个文件夹下存放该标签对应的csv文件
    > 对应的调用方式为 read_csv(r"C:/Users/xxx/Desktop/img", [a, b, c])
    > 返回值为一个dict <str,list> {
                'a': [[...],[...]],
                'b': [[...],[...]],
                'c': [[...],[...]]
        }
    get data from csv file
    :param label_arr:
    :param base_path:
    :return:
    """
    data = {}
    for label in label_arr:
        file_path = "{}.csv".format(os.path.join(os.path.join(base_path, label), label))
        data[label] = np.loadtxt(file_path, dtype=np.uint8, delimiter=",", skiprows=0)
    return data


def build_dataset(data):

    x = []
    y = []
    index = 0
    for label, value in data.items():
        label_num = value.shape[0]
        x.extend(value)
        y.extend(np.ones(label_num)*index)
        index = index + 1
    return np.array(x), np.array(y).flatten()


def train_svm(x, y):
    """
    train a svm model
    :param x: the input signals, 1-D
    :param y: the label
    :return:
    """
    train_data, test_data, train_label, test_label = \
        sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.8, test_size=0.2)
    classifier = svm.SVC(C=2, kernel='linear', gamma=5, decision_function_shape='ovr')  # ovr: one to many
    classifier.fit(train_data, train_label.ravel())
    # joblib.dump(classifier, "svm_model.m")  # you can save the model
    # calculate the acc
    print("training: {}".format(classifier.score(train_data, train_label)))
    print("testing: {}".format(classifier.score(test_data, test_label)))
    y_pre = classifier.predict(test_data)
    print(classification_report(test_label, y_pre))


if __name__ == '__main__':
    # 1. 准备数据 利用matlab生成固定大小的图像，并且建立对应的N个标签文件夹进行存放（N为标签数量）
    # 2. 将每个标签的数据都统计到一个csv文件中，一共会生成N个csv文件
    label_arr = []
    for i in range(0, 26):
        label_arr.append(chr(97+i))
    # print(label_arr)
    base_path = r"D:\AcouInputDataSet\single_img\single_split"
    # img2csv(base_path, label_arr)
    # # # # 3. 从N个csv文件中读取数据，得到一个dict
    data = read_csv(base_path, label_arr)
    # # 4. 生成一个数据集
    # # data = {
    # #     "a": np.random.randn(100, 64*64),
    # #     "b": np.random.randn(100, 64*64),
    # # }
    x, y = build_dataset(data)
    # # 5. 训练数据
    train_svm(x, y)
    # base_path = r"D:\AcouInputDataSet\single_img\single"
    # target_path = r"D:\AcouInputDataSet\single_img\single_split"
    # if not os.path.exists(target_path):
    #     os.mkdir(target_path)
    # for root, dirs, files in os.walk(base_path):
    #     for file in files:
    #         label = file.split("_")[0]
    #         target_folder = os.path.join(target_path, label)
    #         if not os.path.exists(target_folder):
    #             os.mkdir(target_folder)
    #         shutil.move(os.path.join(base_path, file), target_folder)
