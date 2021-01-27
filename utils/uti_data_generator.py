# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
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
import random
import time
import numpy as np
import tensorflow as tf
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

    FEATURE_WINDOW_SIZE = config_all["FEATURE_WINDOW_SIZE"]

# Own modules
# from {path} import {class}
# […]
class Data_Generator(object):


    def __init__(self, data_path, batch_size):

        self._data_path = data_path
        
        self._batch_size = batch_size

    def get_train_data_sum(self):
        '''Read how many "skeleton images" is in this dataset
        Return:
        data_sum {int}: The summe of dataset
        '''
        with np.load(self._data_path) as data:
            data_sum = data['LABEL_TRAIN'].shape[0]
        return data_sum
    
    def get_test_data_sum(self):
        '''Read how many "skeleton images" is in this dataset
        Return:
        data_sum {int}: The summe of dataset
        '''
        with np.load(self._data_path) as data:
            data_sum = data['LABEL_TEST'].shape[0]
        return data_sum

    def batch_cursors(self, data_sum):
        ''' This function will generate a list of lists, which contains the index of the
            input data for the network. The contains should be like devide the dataset acccording to the batch_size,
            one list contains a batch, and all batchs will be listed. If the sum of input dataset could not be divied by batchsize, 
            fill it with random index.
            Argumtnt: 
            self
            data_sum {int}: The summe of the dataset
            Return:
            batch_output_cursors {list}: The list of lists, if batch_size = 64,this looks like: 
                [[0, ... 63],
                 [64,... 127],
                 ...........]
        '''
        batch_output_cursors = []
        file_remainder = data_sum % self._batch_size 
        file_integer = data_sum // self._batch_size
        file_cursors = np.arange(self._batch_size, data_sum, self._batch_size)
        if 0 == file_remainder:
            file_cursors = np.arange(self._batch_size, data_sum+1, self._batch_size) 

        batch_output_cursors.append(list(range(0, file_cursors[0])))
        for integer_index in range(1, file_integer):
            batch_output_cursors.append(list(range(file_cursors[integer_index - 1], file_cursors[integer_index])))
        assert (len(batch_output_cursors) == file_integer)
        if file_remainder != 0:
            generate_num = self._batch_size - file_remainder
            generate_num_cursors = list(random.sample(range(0, data_sum), generate_num))
            final_cursor = list(range(file_cursors[-1], data_sum))
            final_cursors = final_cursor + generate_num_cursors
            batch_output_cursors.append(final_cursors)
            # return a list with 
        return batch_output_cursors

    def generate_batch_data(self, train_data_index, single_batch_cursors):
        batch_data = []
        batch_labels = []
        batch_diff_data = []
        for cursor in single_batch_cursors:
            data_index = train_data_index[cursor]

           
            output_data, diff_output_data, label = self.get_single_data(data_index=data_index)

            batch_data.append(output_data)
            batch_diff_data.append(diff_output_data)
            batch_labels.append(label)

        batch_data = np.array(batch_data, dtype='float32')
        batch_diff_data = np.array(batch_diff_data, dtype='float32')
        batch_labels = np.reshape(batch_labels, (1, batch_size))
        return batch_data, batch_diff_data, batch_labels

    def get_single_data(self, data_index):
        with np.load(self._data_path) as data:
            datasets_position = data['FEATURES_POSITION'][data_index]
            datasets_velocity = data['FEATURES_VELOCITY'][data_index]
            labels = data['FEATURES_LABELS'][data_index]
        return datasets_position, datasets_velocity, labels
    
    def get_single_data_v1(self, data_index, s_p, s_v, label):
        
        datasets_position = s_p[data_index]
        datasets_velocity = s_v[data_index]
        labels = label[data_index]
        return datasets_position, datasets_velocity, labels

    def generate_batch_data_v1(self, train_data_index, single_batch_cursors, s_p, s_v, label):
        batch_data = []
        batch_labels = []
        batch_diff_data = []
        for cursor in single_batch_cursors:
            data_index = train_data_index[cursor]

           
            output_data, diff_output_data, output_label = self.get_single_data_v1(data_index=data_index, s_p=s_p, s_v=s_v, label=label)

            batch_data.append(output_data)
            batch_diff_data.append(diff_output_data)
            batch_labels.append(output_label)

        batch_data = np.array(batch_data, dtype='float32')
        batch_diff_data = np.array(batch_diff_data, dtype='float32')
        batch_labels = np.array(batch_labels, dtype='i')
     
        from keras.utils import to_categorical
        batch_labels_bi = to_categorical(batch_labels, num_classes=5)
        return batch_data, batch_diff_data, batch_labels_bi
    
    def get_test_single_data(self, data_index, test_pos, test_vol, test_label):
        test_position = test_pos[data_index]
        test_velocity = test_vol[data_index]
        test_labels = test_label[data_index]
        return test_position, test_velocity, test_labels

    def get_test_batch(self, test_data_index, single_batch_cursors, test_pos, test_vol, test_label):
        test_batch_data = []
        test_batch_labels = []
        test_batch_diff_data = []    

        for cursor in single_batch_cursors:
            data_index = test_data_index[cursor]

           
            output_data, diff_output_data, output_label = self.get_test_single_data(data_index=data_index, test_pos=test_pos, test_vol=test_vol, test_label=test_label)

            test_batch_data.append(output_data)
            test_batch_diff_data.append(diff_output_data)
            test_batch_labels.append(output_label)

        test_batch_data = np.array(test_batch_data, dtype='float32')
        test_batch_diff_data = np.array(test_batch_diff_data, dtype='float32')
        test_batch_labels = np.array(test_batch_labels, dtype='i')

        from keras.utils import to_categorical
        test_batch_labels_bi = to_categorical(test_batch_labels, num_classes=5)
        return test_batch_data, test_batch_diff_data, test_batch_labels_bi


def load_data(data_path):
    with np.load(data_path) as data:
        datasets_position = data['FEATURES_POSITION']
        datasets_velocity = data['FEATURES_VELOCITY'] 
        labels = data['FEATURES_LABELS']
    return datasets_position, datasets_velocity, labels

def main_temp():
    indices = np.random.permutation(datasets_position.shape[0])
    valid_cnt = int(datasets_position.shape[0] * 0.3)
    test_idx,training_idx=indices[:valid_cnt],indices[valid_cnt:]
    test, train = datasets_position[test_idx,:], datasets_position[training_idx,:]
    test_labels, train_labels = labels[test_idx], labels[training_idx]
    test_vol, train_vol = datasets_velocity[test_idx,:], datasets_velocity[training_idx]

def pack_reshaped_data(self):
    datasets_position_dir = []
    datasets_velocity_dir = []
    label_dir = []
    data_p, data_v, labels = self.load_data()
    for ind in range(len(labels)):
        datasets_position, datasets_velocity, label = self.reshape_single_dataset(data_p[ind], data_v[ind], labels[ind])
        datasets_position_dir.append(datasets_position)
        datasets_velocity_dir.append(datasets_velocity)
        label_dir.append(label)
    return datasets_position_dir, datasets_velocity_dir, label_dir

if __name__ == '__main__':
    pass
