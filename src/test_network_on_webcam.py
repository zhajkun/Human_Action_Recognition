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

# […]

# Libs

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

    VIDEO_PATH = 'data_test/exercise.avi'
    # output

def devide_scale():
    pass

def extract_skeletons_in_dict(human_ids, dict_src):
    lists_dirs = []
    for i in range(len(human_ids)):
        list_dir = di


def main_function():

    # initialize the frames counter at -1, so the first incomming frames is 0
    iFrames_Counter = -1
    # initialize the skeleton detector
    skeleton_detector = uti_openpose.Skeleton_Detector(OPENPOSE_MODEL, OPENPOSE_IMAGE_SIZE)

    # load the trained two stream model
    Nerwork = tf.keras.models.load_model(MODEL_PATH)

    # select the data source
    images_loader = uti_images_io.Read_Images_From_Video(VIDEO_PATH)
    # images_loader = uti_images_io.Read_Images_From_Webcam(10, 0)

    # initialize the skeleton detector   
    Images_Displayer = uti_images_io.Image_Displayer()
    
    # initialize the skeleton detector
    Featurs_Generator_0 = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)
    Featurs_Generator_1 = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)
    Featurs_Generator_2 = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)
    Featurs_Generator_3 = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)
    Featurs_Generator_4 = uti_features_extraction.Features_Generator(FEATURE_WINDOW_SIZE)

    # initialize Multiperson Tracker
    Local_Tracker = uti_tracker.Tracker()

    #################################################################################################
    prev_human_ids = []
    prediction_history = []
    predict_scores_0 = []
   
    positions_temp = []
    velocity_temp = []
    prev_skeletons = []

    invalid_skeletons_counter = 0

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

###########################################################################################################################################
        # sort and track humans in frames
        skeletons_dict = Local_Tracker.track(skeletons_lists)
        
        # get human ids and skeletons seperatly
        human_ids, skeletons_tracked_lists = map(list, zip(*skeletons_dict.items()))

        skeletons_tracked_lists = uti_features_extraction.rebuild_skeleton_joint_order(skeletons_tracked_lists)

        images_display = images_src.copy()

        Images_Displayer.display(images_display)

        if not prev_human_ids and human_ids:

            prev_human_ids = human_ids



if __name__ == '__main__':
    main_function()
 