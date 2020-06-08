# Version

"""
{
    Data Generator for training model, the first version of this model should generate a abtch of 3D numpy
    arraies as the input of this network, remeber to convert the order of the dataset first
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import json
import numpy as np
import random
# […]

# Libs
# import pandas as pd # Or any other
# […]
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)


with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["train.py"]
    FEATURE_WINDOW_SIZE = config_all["FEATURE_WINDOW_SIZE"]
    config = config_all["train.py"]
    FEATURES_SRC = config["input"]["FEATURES"]
    sys.path.append(ROOT)
# Own modules
# from {path} import {class}
# […]
def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path
class DataGenerator(object):

    def __init__(self,FEATURES_SRC, batch_size=64, frames=30):
        self._FEATURES_SRC = FEATURES_SRC
        self._batch_size = batch_size
        self._frames = frames

    # def get_data_name(self):
    #     # 获取hdf5文件中的data set名字，返回一个list
    #     with h5py.File(self._h5file_path, 'r') as h5file:
    #         return [data_name for data_name in h5file]

    def get_cursors(self):
        with np.load(self._FEATURES_SRC) as data:
            datasets_position = data['FEATURES_POSITION']
            datasets_velocity = data['FEATURES_VELOCITY'] 
            labels = data['FEATURES_LABELS']
            data_frames = len(labels)
        # 生成30帧随机帧 先读取有多少个data， 再除以30， 
        output_cursors = []
        add_num = 1
        remainder = data_frames % self._frames #余数
        integer = data_frames // self._frames #商
        # 这里的+1是为了取末尾的index，-1是因为list索引从0开始
        max_num = integer * (self._frames + 1) - 1
        min_num = integer * 1 - 1
        # data_cursors每一段数据的结束位置
        data_cursors = np.arange(min_num, max_num, integer)

        if remainder != 0:
            # random.sample()生成一个在一定范围内不重复的list
            # extra_cursors增加的数据的索引
            extra_cursors = sorted(random.sample(range(self._frames), remainder))

            # 改变一个位置的index其以后的所有都要改变
            for add_index in extra_cursors:
                # 这里由于data_cursors是一个numpy array所以利用broadcast性质
                data_cursors[add_index:] = data_cursors[add_index:] + add_num
            assert (data_cursors[-1] == data_frames - 1)

        output_cursors.append(random.randint(0, data_cursors[0]))
        for cursor_index in range(1, self._frames):
            output_cursors.append(random.randint(
                data_cursors[cursor_index - 1] + 1, data_cursors[cursor_index]))
        # print(output_cursors)
            # 返回帧数的选择list
        return output_cursors

    def reshape_dataset(list_src):
        iFrames = 10
        iJoints = 35
        iDimenssion = 2
        zero_metric = np.zeros([1, 35, 2], dtype=float)

        if(len(list_src)>699):
            list_dir = np.reshape(list_src, (iFrames, iJoints, iDimenssion))
            list_dir = np.expand_dims(list_dir, axis=0)
        else:
            list_dir = np.reshape(list_src, (iFrames-1, iJoints, iDimenssion))
            list_dir = np.append(list_dir, zero_metric, axis=0)
            list_dir = np.expand_dims(list_dir, axis=0)
        return list_dir

    def get_single_data(self, data_name, output_cursors, body_id):
        # body_id是取primary和secondary的时候用的，当4维数据取出期中一维的某个则变成了三维数据
        zero_metric = np.zeros([1, 25, 3], dtype=float)
        with h5py.File(self._h5file_path, 'r') as h5file:
            data = h5file[data_name]
            output_data = data[body_id, output_cursors, :, :]
            output_data = np.array(output_data, dtype='float32')
            # 计算帧差 np.diff()arr,n是计算几次差值，axis在某个维度计算差值你

            diff_output_data = np.diff(output_data, n=1, axis=0)
            # np.append()将value在维度axis=的附在arr后
            diff_output_data = np.array(np.append(diff_output_data, zero_metric, axis=0), dtype='float32')
            # 这里的label（60，1）还是（1，60）？？
            label = np.array(data.attrs['label']).reshape((60,))
            assert (output_data.shape == (self._frames, 25, 3))
            # assert(label.shape == (60, 1))
        return output_data, diff_output_data, label

    # # 与训练集不同测试集的batch size为1所以需要扩展维度
    # def get_tst_single_data(self, data_name, output_cursors, body_id):
    #     zero_metric = np.zeros([1, 25, 3], dtype=float)
    #     with h5py.File(self._h5file_path, 'r') as h5file:
    #         data = h5file[data_name]
    #         output_data = data[body_id, output_cursors, :, :]
    #         output_data = np.array(output_data, dtype='float32')
    #         # 计算帧差 np.diff()arr,n是计算几次差值，axis在某个维度计算差值你
    #         diff_output_data = np.diff(output_data, n=1, axis=0)
    #         # np.append()将value在维度axis=的附在arr后
    #         diff_output_data = np.array(np.append(diff_output_data, zero_metric, axis=0), dtype='float32')
    #         output_data = np.expand_dims(output_data, axis=0)
    #         diff_output_data = np.expand_dims(diff_output_data, axis=0)
    #         # 需要的是（1，60）的向量所以这里进行转置
    #         label = np.array(data.attrs['label']).T
    #         # assert(output_data.shape == (self._frames, 25, 3))
    #         # assert(label.shape == (60, 1))
    #     return output_data, diff_output_data, label

    def batch_cursors(self, data_num):
        # 数据需不需要打乱（个人感觉不是很需要）
        # 根据训练集和测试集以及batch size，生成每一个batch对应的索引编号
        # 这里与生成单个文件的cursors相似但不是很相同，不用考虑时序每个个体是独立的且要遍历完一遍
        batch_output_cursors = []
        file_remainder = data_num % self._batch_size
        file_integer = data_num // self._batch_size
        file_cursors = np.arange(self._batch_size, data_num, self._batch_size)

        batch_output_cursors.append(list(range(0, file_cursors[0])))
        for integer_index in range(1, file_integer):
            batch_output_cursors.append(list(range(file_cursors[integer_index - 1], file_cursors[integer_index])))
        assert (len(batch_output_cursors) == file_integer)
        if file_remainder != 0:
            generate_num = self._batch_size - file_remainder
            generate_num_cursors = list(random.sample(range(0, data_num), generate_num))
            final_cursor = list(range(file_cursors[-1], data_num))
            final_cursors = final_cursor + generate_num_cursors
            batch_output_cursors.append(final_cursors)
            # 返回一个 list 含有 file_integer+1 个list
        return batch_output_cursors

    # # 分两次生成不过只用一次的label
    # def generate_batch_data(self, name_list, single_batch_cursors, body_id):
    #     batch_data = []
    #     batch_labels = []
    #     batch_diff_data = []
    #     for cursor in single_batch_cursors:
    #         data_name = name_list[cursor]

    #         output_cursors = self.get_cursors(data_name=data_name)
    #         output_data, diff_output_data, label = self.get_single_data(data_name=data_name,
    #                                                                     output_cursors=output_cursors, body_id=body_id)

    #         batch_data.append(output_data)
    #         batch_diff_data.append(diff_output_data)
    #         batch_labels.append(label)

    #     batch_data = np.array(batch_data, dtype='float32').reshape((self._batch_size, self._frames, 25, 3))
    #     batch_diff_data = np.array(batch_diff_data, dtype='float32').reshape((self._batch_size, self._frames, 25, 3))
    #     batch_labels = np.array(batch_labels, dtype='float32')
    #     return batch_data, batch_diff_data, batch_labels



if __name__ == '__main__':
    data = DataGenerator("C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features.npz",32,10)
    # namelist = data.get_data_name()
    # output=data.get_cursors(namelist[0])
    # outputdata, diffoutput, label=data.get_single_data(namelist[0], output, 0)
    # print(outputdata)
    # print(diffoutput)
    # print(label)
    a = data.get_cursors()
    print(a)