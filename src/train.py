# temporal version, for one stream only
import numpy as np
import json
import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
# from tensorflow.keras.layers import Conv2D

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.uti_images_io as uti_images_io

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings

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

def main_functions():
    img_loader = uti_images_io.Read_Images_From_Webcam(10,0)
    img_displayer = uti_images_io.Image_Displayer()

    while img_loader.Image_Captured:
        img_src = img_loader.Read_Image()
        img_dst = img_src.copy()
        img_displayer.display(img_dst)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    img_loader.Stop()

if __name__ == "__main__":
    main_functions()
