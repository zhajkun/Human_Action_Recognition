# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

'''
{
    first version of two-stream network
}
{License_info}
'''

# Futures

# […]

# Built-in/Generic Imports
import sys
import os
import numpy as np
import json
import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as pl
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
if True:  # Include project path
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+'/'
    sys.path.append(ROOT)
    import utils.uti_data_generator as uti_data_generator
    import utils.uti_commons as uti_commons
    import utils.uti_images_io as uti_images_io
# […]

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != '/') else path

# -- Settings

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all['train.py']

    # common settings

    CLASSES = np.array(config_all['classes'])
    IMAGE_FILE_NAME_FORMAT = config_all['IMAGE_FILE_NAME_FORMAT']
    SKELETON_FILE_NAME_FORMAT = config_all['SKELETON_FILE_NAME_FORMAT']
    IMAGES_INFO_INDEX = config_all['IMAGES_INFO_INDEX']
    FEATURE_WINDOW_SIZE = config_all['FEATURE_WINDOW_SIZE'] 
    JOINTS_NUMBER = config_all['JOINTS_NUMBER']
    CHANELS = config_all['CHANELS']


    # input

    # output

epochs = 100
BATCH_SIZE = 64
input_shape = (FEATURE_WINDOW_SIZE, JOINTS_NUMBER, CHANELS)
use_bias = True
graph_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/model.png'
train_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_train.npz'
test_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_test.npz'
save_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/model/two_stream.h5'
# -- Function
def load_single_train_npy(sFile_path, data_index):
    with np.load(sFile_path) as data:
        datasets_position = data['POSITION_TRAIN'][data_index]
        datasets_velocity = data['VELOCITY_TRAIN'][data_index]
        labels = data['LABEL_TRAIN'][data_index]
    return datasets_position, datasets_velocity, labels
if __name__ == '__main__':
    train_path = "C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_train.npz"
    new_model = tf.keras.models.load_model(save_path)
    new_model.summary()
    acc_corr = 0
    for i in range(1000):
        pos, vel, label = load_single_train_npy(train_path, i)
        pos = np.array(pos, dtype=float)
        vel = np.array(vel, dtype=float)
        # label = np.array(label, dtype=int)
        up_0 = np.expand_dims(pos, axis=0)
        up_1 = np.expand_dims(vel, axis=0)
        down_0 = np.expand_dims(pos, axis=0)
        down_1 = np.expand_dims(vel, axis=0)
        # label = np.expand_dims(label, axis=0)
        # from keras.utils import to_categorical
       
        # label = to_categorical(label, num_classes=5)
        # label = np.expand_dims(label, axis=0)
        # loss, acc = new_model.evaluate([up_0, up_1, down_0, down_1], label, verbose=2)
        prediction = new_model.predict([up_0, up_1, down_0, down_1])
        print('Predicted: Put in basket:', int(prediction[0][0]))
        print('Predicted: Waving:', int(prediction[0][1]))
        print('Predicted: Standing:', int(prediction[0][2]))
        print('Predicted: Walking:', int(prediction[0][3]))
        print('Predicted: walk to me:', int(prediction[0][4]))
        print('Label:', CLASSES[label])
    
    Web_Cam_Frames = uti_images_io.Read_Images_From_Webcam
