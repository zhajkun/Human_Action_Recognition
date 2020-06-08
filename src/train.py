# temporal version, for one stream only
import numpy as np
import json
import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras.layers import Conv2D

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    # import utils.lib_plot as lib_plot
    # from utils.lib_classifier import ClassifierOfflineTrain



def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["train.py"]

    # common settings

    CLASSES = np.array(config_all["classes"])
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    SKELETON_FILE_NAME_FORMAT = config_all["SKELETON_FILE_NAME_FORMAT"]
    IMAGES_INFO_INDEX = config_all["IMAGES_INFO_INDEX"]

        # openpose



    # input
    FEATURES_SRC = par(config["input"]["FEATURES"])
    # output
    
    MODEL_PATH = par(config["output"]["MODEL_PATH"])

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

# -- Function
def load_datasets():    
    with np.load(FEATURES_SRC) as data:
        datasets_position = data['FEATURES_POSITION']
        datasets_velocity = data['FEATURES_VELOCITY'] 
        labels = data['FEATURES_LABELS']
    return datasets_position, datasets_velocity, labels

def split_dataset(datasets_position, datasets_velocity, labels):
    #  if datasets_position
    return True

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

def main():
    # split the datasets, 705 for training, 30% for test
    datasets_position, datasets_velocity, labels = load_datasets()
    indices = np.random.permutation(datasets_position.shape[0])
    valid_cnt = int(datasets_position.shape[0] * 0.3)
    test_idx,training_idx=indices[:valid_cnt],indices[valid_cnt:]
    test, train = datasets_position[test_idx,:], datasets_position[training_idx,:]
    test_labels, train_labels = labels[test_idx], labels[training_idx]
    test_vol, train_vol = datasets_velocity[test_idx,:], datasets_velocity[training_idx]

    input_shape=(700, 1)
    l2 = tf.keras.regularizers.l2(l=0.001)
    model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu')
        ])
    # model = tf.keras.Sequential([
    #     # tf.keras.layers.Conv1D(filters=64, kernel_size=(3), strides=(1), padding='valid',
    #     #            use_bias=True),
    #     tf.keras.layers.Dense(256, activation='relu', use_bias=True, kernel_regularizer=l2),
    #     # tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(128, activation='relu', use_bias=True),
    #     tf.keras.layers.Dense(96, activation='relu', use_bias=True),
    #     tf.keras.layers.Dense(units=5, activation='softmax', use_bias=True)
    #     ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test,  test_labels, verbose=2)
    model.summary()

    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    datasets_position, datasets_velocity, labels = load_datasets()
    # print(datasets_velocity[1])
    print(datasets_position[1])
    # print(labels[1])
    # check_diff = np.diff(datasets_position, n=1, axis=1)
    test_reshape_pos = reshape_dataset(datasets_position[1])
    test_reshape_vel = reshape_dataset(datasets_velocity[1])
    
    print(test_reshape_pos.shape)
    print(test_reshape_vel.shape)
    print("Finish")