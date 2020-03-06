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

    import utils.lib_plot as lib_plot
    from utils.lib_classifier import ClassifierOfflineTrain



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


def main():
    datasets_position, datasets_velocity, labels = load_datasets()

    datasets_position = np.random.rand(100, 5)
    np.random.shuffle(datasets_position)
    np.random.shuffle(labels)
    training, test = datasets_position[:80,:], datasets_position[80:,:]
    train_labels, test_labels = labels[:80,:], labels[80:,:]


    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
        ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(training, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    main()
    print("Finish")