import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_single_train_npy(sFile_path, data_index):
    with np.load(sFile_path) as data:
        datasets_position = data['POSITION_TRAIN'][data_index]
        datasets_velocity = data['VELOCITY_TRAIN'][data_index]
        labels = data['LABEL_TRAIN'][data_index]
    return datasets_position, datasets_velocity, labels

def load_single_test_npy(sFile_path, data_index):
    with np.load(sFile_path) as data:
        datasets_position = data['POSITION_TEST'][data_index]
        datasets_velocity = data['VELOCITY_TEST'][data_index]
        labels = data['LABEL_TEST'][data_index]
    return datasets_position, datasets_velocity, labels

def plot_skeletons(skeletons_src):
    skeletons_x = []
    skeletons_y = []
    fig = plt.figure()
    for i in range(10):

        skeletons_x = skeletons_src[i,:,0]
        skeletons_y = skeletons_src[i,:,1]
        plt.plot(skeletons_x, skeletons_y, 'rx')
        plt.xlim(0,1)
        plt.ylim(1,0)  
    plt.show(block=False)
    plt.pause(1)
    plt.close()
if __name__ == "__main__":
    train_path = "C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_train.npz"
    test_path = "C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_test.npz"
    CLASSES = ["PUTINBASKET","WAVING","STANDING","WALKING","WALKTOME"]

   
    pos, vel, label = load_single_train_npy(train_path, 9999)
    label_int = int(label)
    print(CLASSES[label_int])
    plot_skeletons(pos)
  