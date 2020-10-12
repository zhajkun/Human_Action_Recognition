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
import matplotlib.pylab as pl

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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
# […]

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != '/') else path

# -- Settings

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all['train_network.py']

    # common settings

    ACTION_CLASSES = np.array(config_all['ACTION_CLASSES'])
    IMAGE_FILE_NAME_FORMAT = config_all['IMAGE_FILE_NAME_FORMAT']
    SKELETON_FILE_NAME_FORMAT = config_all['SKELETON_FILE_NAME_FORMAT']
    IMAGES_INFO_INDEX = config_all['IMAGES_INFO_INDEX']
    FEATURE_WINDOW_SIZE = config_all['FEATURE_WINDOW_SIZE'] 
    JOINTS_NUMBER = config_all['JOINTS_NUMBER']
    CHANELS = config_all['CHANELS']
    BATCH_SIZE = config_all['BATCH_SIZE']
    EPOCHS = config_all['EPOCHS']

    # input

    FEATURES_TRAIN = par(config['input']['FEATURES_TRAIN'])
    FEATURES_TEST = par(config['input']['FEATURES_TEST'])

    # output
    
    MODEL_PATH = par(config['output']['MODEL_PATH'])
    TXT_FILE_PATH = config['output']['TXT_FILE_PATH']
    FIGURE_PATH = config['output']['FIGURE_PATH']

input_shape = (FEATURE_WINDOW_SIZE, JOINTS_NUMBER, CHANELS)
use_bias = True

# -- Function
def load_train_datasets(feature_path):
    with np.load(feature_path) as data:
        train_position = data['POSITION_TRAIN']
        train_velocity = data['VELOCITY_TRAIN'] 
        train_labels = data['LABEL_TRAIN']
    return train_position, train_velocity, train_labels

def load_test_datasets(feature_path):
    with np.load(feature_path) as data:
        test_position = data['POSITION_TEST']
        test_velocity = data['VELOCITY_TEST']
        test_labels = data['LABEL_TEST']
    return test_position, test_velocity, test_labels

def shared_stream(x_shape):

    inputs = tf.keras.Input(shape=x_shape)

    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(inputs)

    conv1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                   use_bias=use_bias)(conv1)
    
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    
    conv2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)

    conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                use_bias=use_bias)(conv2)
    
    conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', 
                    use_bias=use_bias)(conv3)

    conv3 = tf.keras.layers.Activation('relu')(conv3)
    
    conv3 = tf.keras.layers.Dropout(0.5)(conv3)
    
    conv3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)

    outputs = conv3

    shared_layers = tf.keras.Model(inputs, outputs)

    return shared_layers

def model():
    up_0 = tf.keras.layers.Input(shape=input_shape, name='up_stream_0')
    up_1 = tf.keras.layers.Input(shape=input_shape, name='up_stream_1')
    down_0 = tf.keras.layers.Input(shape=input_shape, name='down_stream_0')
    down_1 = tf.keras.layers.Input(shape=input_shape, name='down_stream_1')

    up_stream = shared_stream(x_shape=input_shape)
    down_stream = shared_stream(x_shape=input_shape)

    up_feature_0 = up_stream(up_0)
    up_feature_1 = up_stream(up_1)
    down_feature_0 = down_stream(down_0)
    down_feature_1 = down_stream(down_1)

    ###only for 1 frame use
    # up_feature_0 = up_0
    # up_feature_1 = up_1
    # down_feature_0 = down_0
    # down_feature_1 = down_1

    # Flatten all 4 streams
    up_feature_0 = tf.keras.layers.Flatten()(up_feature_0)
    up_feature_1 = tf.keras.layers.Flatten()(up_feature_1)
    down_feature_0 = tf.keras.layers.Flatten()(down_feature_0)
    down_feature_1 = tf.keras.layers.Flatten()(down_feature_1)
    
    # only use the maximun features for the next layer
    up_feature = tf.keras.layers.Maximum()([up_feature_0, up_feature_1])
    down_feature = tf.keras.layers.Maximum()([down_feature_0, down_feature_1])
   
    # concate the features from position and velocity
    feature = tf.keras.layers.concatenate([up_feature, down_feature])

    fc_1 = tf.keras.layers.Dense(units=256, activation='relu', use_bias=True, kernel_regularizer=l2(0.001))(feature)
    fc_1 = tf.keras.layers.Dropout(0.5)(fc_1)

    fc_2 = tf.keras.layers.Dense(units=128, activation='relu', use_bias=True)(fc_1)

    fc_3 = tf.keras.layers.Dense(units=64, activation='relu', use_bias=True)(fc_2)

    fc_4 = tf.keras.layers.Dense(units=32, activation='relu', use_bias=True)(fc_3)

    fc_5 = tf.keras.layers.Dense(units=16, activation='relu', use_bias=True)(fc_4)

    fc_6 = tf.keras.layers.Dense(units=len(ACTION_CLASSES), activation='softmax', use_bias=True)(fc_5) # units=len(ACTION_CLASSES)

    network = tf.keras.Model(inputs=[up_0, up_1, down_0, down_1], outputs=fc_6)
    return network

def train_model_on_batch_v1(network):
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # original setup from paper
    network.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    network.summary()

    # network.load_weights(weight_path)

    # tf.keras.utils.plot_model(network, to_file= FIGURE_PATH + 'Model.png')

    batch_num = 0
    model_save_acc = 0
    all_train_accuracy = []
    all_train_loss = []
    all_tst_accuracy = []
    all_tst_loss = []

    # load dataset to memory, temp.
    train_position, train_velocity, train_labels = load_train_datasets(FEATURES_TRAIN)
    test_position, test_velocity, test_labels = load_test_datasets(FEATURES_TEST)

    train_data = uti_data_generator.Data_Generator(FEATURES_TRAIN, BATCH_SIZE)
    train_data_sum = train_data.get_train_data_sum()
    train_data_index = np.arange(0, train_data_sum)
    train_data_cursors = train_data.batch_cursors(train_data_sum)
    index_num = len(train_data_cursors)

    # test_position = np.expand_dims(test_position, axis=0)
    # test_velocity = np.expand_dims(test_velocity, axis=0)
    test_data = uti_data_generator.Data_Generator(FEATURES_TEST, BATCH_SIZE)
    test_data_sum = test_data.get_test_data_sum()
    test_data_index = np.arange(0, test_data_sum)
    test_data_cursors = test_data.batch_cursors(test_data_sum)
    test_index_num = len(test_data_cursors)

    for epoch in range(EPOCHS):
        accuracy_list = []
        loss_list = []
        test_accuracy_list = []
        test_loss_list = []
        print(epoch + 1, ' epoch is beginning......')
        '''
        '''
        for ind in range(index_num):
            batch_num += 1
            up_data_0, down_data_0, train_labels_0 \
                = train_data.generate_batch_data_v1(train_data_index, train_data_cursors[ind], train_position, train_velocity, train_labels)
            up_data_1, down_data_1, train_labels_1 \
                = train_data.generate_batch_data_v1(train_data_index, train_data_cursors[ind], train_position, train_velocity, train_labels)
            train_loss = network.train_on_batch([up_data_0, up_data_1, down_data_0, down_data_1], train_labels_0)
            accuracy_list.append(train_loss[1])
            loss_list.append(train_loss[0])
            if batch_num % 50 == 0:
                print('the {:03d} batch: loss: {:.3f}  accuracy: {:.3%}'.format(batch_num, train_loss[0], train_loss[1]))

        epoch_accuracy = sum(accuracy_list) / len(accuracy_list)
        epoch_loss = sum(loss_list) / len(loss_list)
        all_train_accuracy.append(epoch_accuracy)
        all_train_loss.append(epoch_loss)

        print('the {:03d} epoch: mean loss: {:.3f}    mean accuracy: {:.3%}'.format(epoch + 1, epoch_loss, epoch_accuracy))



        if epoch >= 1:
            tst_accuracy_list = []
            tst_loss_list = []
            for num in range(test_index_num):
                tst_up_0, tst_down_0, tst_labels_0 \
                    = test_data.get_test_batch(test_data_index, test_data_cursors[num], test_position, test_velocity, test_labels)
                tst_up_1, tst_down_1, tst_labels_1 \
                    = test_data.get_test_batch(test_data_index, test_data_cursors[num], test_position, test_velocity, test_labels)
                tst_loss = network.test_on_batch([tst_up_0, tst_up_1, tst_down_0, tst_down_1], tst_labels_0)
                
                # test_result = network.evaluate([tst_up_0, tst_up_1, tst_down_0, tst_down_1], tst_labels_0)

                
                tst_loss_list.append(tst_loss[0])
                tst_accuracy_list.append(tst_loss[1])
            tst_accuracy = sum(tst_accuracy_list) / len(tst_accuracy_list)
            tst_loss_output = sum(tst_loss_list) / len(tst_loss_list)

            all_tst_accuracy.append(tst_accuracy)
            all_tst_loss.append(tst_loss_output)
            print('The test data accuracy: {:.3%}'.format(tst_accuracy))
            if tst_accuracy > model_save_acc:
                network.save(MODEL_PATH)
                model_save_acc = tst_accuracy
                print('Model Saved')

    uti_commons.save_listlist(TXT_FILE_PATH + 'all_train_loss.txt', all_train_loss)
    uti_commons.save_listlist(TXT_FILE_PATH + 'all_train_acc.txt', all_train_accuracy)
    uti_commons.save_listlist(TXT_FILE_PATH + 'all_test_loss.txt', all_tst_loss)
    uti_commons.save_listlist(TXT_FILE_PATH + 'all_test_acc.txt', all_tst_accuracy)
    
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].plot(all_train_loss)

    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].plot(all_train_accuracy)
    plt.savefig(FIGURE_PATH + '30frame_v2_c_train.png')

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Test Metrics')

    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].plot(all_tst_loss)

    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].plot(all_tst_accuracy)
    plt.savefig(FIGURE_PATH + '30frame_v2_c_test.png')

def train_model(network):
    pass

if __name__ == '__main__':
    # start to train the network, change the input shape of network via config/config.json
    time_start = time.time()
    network = model()
    train_model_on_batch_v1(network)
    time_end = time.time()
    print('Finish')
    print('Time Cost:', time_end - time_start, 'seconds' )

