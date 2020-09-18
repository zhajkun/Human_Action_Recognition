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

import argparse

'''
# tf.ConfigProto.gpu_options.allow_growth=True
# tf.ConfigProto.gpu_options.per_process_gpu_memory_fraction=0.4
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
'''

# […]

# Libs
# import pandas as pd # Or any other
# […]
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
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
    config = config_all['test_network.py']

    # common settings

    CLASSES = np.array(config_all['ACTION_CLASSES'])
    IMAGE_FILE_NAME_FORMAT = config_all['IMAGE_FILE_NAME_FORMAT']
    SKELETON_FILE_NAME_FORMAT = config_all['SKELETON_FILE_NAME_FORMAT']
    IMAGES_INFO_INDEX = config_all['IMAGES_INFO_INDEX']
    FEATURE_WINDOW_SIZE = config_all['FEATURE_WINDOW_SIZE'] 
    JOINTS_NUMBER = config_all['JOINTS_NUMBER']
    CHANELS = config_all['CHANELS']
    OPENPOSE_MODEL = config_all['OPENPOSE_MODEL']
    OPENPOSE_IMAGE_SIZE = config_all['OPENPOSE_IMAGE_SIZE']
    # input
    MODEL_PATH = par(config['input']['MODEL_PATH'])
    # output

TEST_FOLDER = 'data_test/UNDEFINED_09-01-13-52-09-823/'
RESULT_LIST = 'home/zhaj/tf_test/Human_Action_Recognition/data_proc/results.txt'
VIDEO_PATH = 'data_test/exercise.avi'

def argument_parser():
    '''
    '''
    parser = argparse.ArgumentParser(
        description='Test Human Action Recognition Model on \n'
        '(1) a video, (2) a folder of images, (3) or web camera.')
    parser.add_argument('-m', '--MODEL_PATH', required=False,
                        default=MODEL_PATH)
    parser.add_argument('-t', '--data_source', required=False, default='Images',
                        choices=['Video', 'Images', 'Webcam'])
    parser.add_argument('-p', '--data_path', required=False, default='data_test/UNDEFINED_08-20-15-49-28-094/',
                        help='path to a video file, or images folder, or webcam. \n'
                        'For video and folder, the path should be '
                        'absolute or relative to this projects root. '
                        'For webcam, either input an index or device name. ')
    args = parser.parse_args()
    return args

def select_data_source(data_source, data_path):

    if data_source == 'Video':
        images_loader = uti_images_io.Read_Images_From_Video(
            data_path,
            sample_interval=3)

    elif data_source == 'Images':
        images_loader = uti_images_io.Read_Images_From_Folder(
            sFolder_Path=data_path)

    elif data_source == 'Webcam':
        if data_path == '':
            webcam_idx = 0
        elif data_path.isdigit():
            webcam_idx = int(data_path)
        else:
            webcam_idx = 0
        images_loader = uti_images_io.Read_Images_From_Webcam(
            10, webcam_idx)
    return images_loader

def devide_scale():
    pass

def main_function():

    # initialize the frames counter at -1, so the first incomming frames is 0
    iFrames_Counter = -1
    # initialize the skeleton detector
    skeleton_detector = uti_openpose.Skeleton_Detector(OPENPOSE_MODEL, OPENPOSE_IMAGE_SIZE)

    # load the trained two stream model
    Nerwork = tf.keras.models.load_model(MODEL_PATH)
    
    args = argument_parser()

    data_source_type = args.data_source
    data_path = args.data_path

    # select the data source
    #
    #
    # images_loader = uti_images_io.Read_Images_From_Webcam(10, 0)
    # images_loader = select_data_source(data_source_type, data_path)
    images_loader = uti_images_io.Read_Images_From_Folder(TEST_FOLDER)
    # images_loader = uti_images_io.Read_Images_From_Video(VIDEO_PATH)
    # initialize the skeleton detector   
    Images_Displayer = uti_images_io.Image_Displayer()
    
    # initialize the skeleton detector
    Featurs_Generator = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)

    #################################################################################################
    
    prediction_history = []
    predict_scores_0 = []
   
    positions_temp = []
    velocity_temp = []
    prev_skeletons = []

    invalid_skeletons_counter = 0
    ##################################################################################################

    while images_loader.Image_Captured():
     
        # iterate the frames counter by 1
        iFrames_Counter += 1
  
        # grab frames from data source
        images_src = images_loader.Read_Image()

        # get detected human(s) from openpose
        humans = skeleton_detector.detect(images_src)

        # convert human(s) to 2d coordinates in a list(of lists)
        skeletons_lists_src, scale_h = skeleton_detector.humans_to_skeletons_list(humans)
        
        # delete invalid skeletons from lists
        skeletons_lists = uti_tracker.delete_invalid_skeletons_from_lists(skeletons_lists_src)

        skeletons_lists = uti_features_extraction.rebuild_skeleton_joint_order(skeletons_lists)

        if not skeletons_lists and not Featurs_Generator._skeletons_deque:
            
            images_display = images_src.copy()

            Images_Displayer.display(images_display)
            
            prediction_history.insert(iFrames_Counter, [0]*5)

            Featurs_Generator._reset()

            continue
        # if Features_generator starts to store skeletons for features, but there were some frames failed, 
        # use the previous skeletons for 
   
        elif not skeletons_lists and Featurs_Generator._skeletons_deque and prev_skeletons:
            
            if invalid_skeletons_counter <= 4:
                # use previous list
                skeletons_lists = prev_skeletons
                
                # increase invalid skeletons counter by 1
                invalid_skeletons_counter += 1

            else:
                
                Featurs_Generator._reset()
                
                invalid_skeletons_counter = 0
                
                prev_skeletons = []

                images_display = images_src.copy()

                Images_Displayer.display(images_display)
                
                prediction_history.insert(iFrames_Counter, [0]*5)

                continue

        prev_skeletons = skeletons_lists
        
        success, features_x, features_xs = Featurs_Generator.calculate_features(skeletons_lists[0])
        
        prediction_history.insert(iFrames_Counter, [0]*5)
        
        if success:  # True if (data length > 5) and (skeleton has enough joints)
        # positions_temp.append(features_x)       
        # velocity_temp.append(features_xs)

            positions_temp = np.array(features_x, dtype=float)
            velocity_temp = np.array(features_xs, dtype=float)
            
            positions_temp = np.expand_dims(positions_temp, axis=0)
            velocity_temp = np.expand_dims(velocity_temp, axis=0)
        
            up_0 = positions_temp
            up_1 = positions_temp
            down_0 = velocity_temp
            down_1 = velocity_temp
        
            prediction = Nerwork.predict([up_0, up_1, down_0, down_1])



            images_display = images_src.copy()

            skeleton_detector.draw(images_display, humans)

            uti_images_io.draw_scores_for_one_person_on_image(images_display, prediction[0])

            skeleton_src = skeletons_lists[0]

            skeleton_src[1::2] = np.divide(skeleton_src[1::2], scale_h)

            images_display = uti_images_io.draw_bounding_box_for_one_person_on_image(images_display, skeleton_src)

            Images_Displayer.display(images_display)

            print(f'\nPredict {iFrames_Counter}th Frame ...')
            print('Predicted: Put in basket: {:.3%}'.format(prediction[0][0]))
            print('Predicted: Sitting: {:.3%}'.format(prediction[0][1]))
            print('Predicted: Standing: {:.3%}'.format(prediction[0][2]))
            print('Predicted: Walking: {:.3%}'.format(prediction[0][3]))
            print('Predicted: Waving: {:.3%}'.format(prediction[0][4]))

            prediction_history[iFrames_Counter]  = prediction[0].tolist()

        else:

            images_display = images_src.copy()

            skeleton_detector.draw(images_display, humans)

            skeleton_src = skeletons_lists[0]

            skeleton_src[1::2] = np.divide(skeleton_src[1::2], scale_h)

            images_display = uti_images_io.draw_bounding_box_for_one_person_on_image(images_display, skeleton_src)

            Images_Displayer.display(images_display)

        predict_scores_0.append(prediction_history[iFrames_Counter])

    uti_commons.save_listlist('data_proc/prediction_0.txt', predict_scores_0)
        # uti_commons.save_listlist('data_proc/prediction_1.txt', predict_scores_1)
        # uti_commons.save_listlist('data_proc/prediction_2.txt', predict_scores_2)
        # uti_commons.save_listlist('data_proc/prediction_3.txt', predict_scores_3)
        # uti_commons.save_listlist('data_proc/prediction_4.txt', predict_scores_4)

if __name__ == '__main__':
    main_function()
 