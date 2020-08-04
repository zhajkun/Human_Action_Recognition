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
import argparse
import cv2
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
    import utils.uti_openpose as uti_openpose
    import utils.uti_features_extraction as uti_features_extraction
    import utils.uti_filter as uti_filter
    import utils.uti_tracker as uti_tracker
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
    OPENPOSE_MODEL = config_all['OPENPOSE_MODEL']
    OPENPOSE_IMAGE_SIZE = config_all['OPENPOSE_IMAGE_SIZE']
    # input

    # output

input_shape = (FEATURE_WINDOW_SIZE, JOINTS_NUMBER, CHANELS)
use_bias = True
graph_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/model.png'
train_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_train.npz'
test_path = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_test.npz'
MODEL_PATH = 'C:/Users/Kun/tf_test/Human_Action_Recognition/model/two_stream.h5'
TEST_FOLDER = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data/Data_Images_10FPS/WALKTOME_02-06-17-27-33-537/'
RESULT_LIST = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/results.txt'
def argument_parser():
    '''
    '''
    parser = argparse.ArgumentParser(
        description="Test Human Action Recognition Model on \n"
        "(1) a video, (2) a folder of images, (3) or web camera.")
    parser.add_argument("-m", "--MODEL_PATH", required=False,
                        default=MODEL_PATH)
    parser.add_argument("-t", "--data_type", required=False, default='webcam',
                        choices=["video", "folder", "webcam"])
    parser.add_argument("-p", "--data_path", required=False, default="",
                        help="path to a video file, or images folder, or webcam. \n"
                        "For video and folder, the path should be "
                        "absolute or relative to this project's root. "
                        "For webcam, either input an index or device name. ")
    args = parser.parse_args()
    return args

def matain_list_size(lists_src):
    empty_frames = [0]*36
    if not lists_src:
        return empty_frames
    elif len(lists_src) != 1:
        return lists_src[0] 

def main_function():

    # load the trained model
    Nerwork = tf.keras.models.load_model(MODEL_PATH)
    # initialize the skeleton detector
    skeleton_decetor = uti_openpose.Skeleton_Detector(OPENPOSE_MODEL, OPENPOSE_IMAGE_SIZE)
    
    # images_loader = uti_images_io.Read_Images_From_Webcam(10, 0)
    images_loader = uti_images_io.Read_Images_From_Folder(TEST_FOLDER)
    
    Images_Displayer = uti_images_io.Image_Displayer()
    Featurs_Generator = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)
    
    predict_scores = []
    
    

    while images_loader.Image_Captured():
        # iFrames_Counter += 1
        positions_temp = []
        velocity_temp = []
        images_src = images_loader.Read_Image()

     
        humans = skeleton_decetor.detect(images_src)
        skeletons_lists, scale_h = skeleton_decetor.humans_to_skeletons_list(humans)
        # skeletons_lists = matain_list_size(skeletons_lists)

        images_display = images_src.copy()  
        skeleton_decetor.draw(images_display, humans)

        images_display = uti_images_io.add_border_to_images(images_display)
        
        Images_Displayer.display(images_display)
        skeletons_lists = uti_tracker.delete_invalid_skeletons_from_dict(skeletons_lists)
        if cv2.waitKey(1) == 27:
            break
        if not skeletons_lists:
            continue
        else:
            success, features_x, features_xs = Featurs_Generator.calculate_features(skeletons_lists)
            if success:  # True if (data length > 5) and (skeleton has enough joints)
                positions_temp.append(features_x)       
                velocity_temp.append(features_xs)

                positions_temp = np.array(positions_temp, dtype=float)
                velocity_temp = np.array(velocity_temp, dtype=float)
            
                up_0 = positions_temp
                up_1 = velocity_temp
                down_0 = positions_temp
                down_1 = velocity_temp
                
                prediction = Nerwork.predict([up_0, up_1, down_0, down_1])
                
        
                print('Predicted: Put in basket: {:.3%}'.format(prediction[0][0]))
                print('Predicted: Waving: {:.3%}'.format(prediction[0][1]))
                print('Predicted: Standing: {:.3%}'.format(prediction[0][2]))
                print('Predicted: Walking: {:.3%}'.format(prediction[0][3]))
                print('Predicted: walk to me: {:.3%}'.format(prediction[0][4]))

if __name__ == '__main__':
    main_function()
