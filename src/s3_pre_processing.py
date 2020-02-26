# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Load the numpy array and pre-processing it
    including rebuild the joint order 
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import json
import numpy as np
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own module
# […]

if True:  # Include project path
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.uti_images_io as uti_images_io
    import utils.uti_openpose as uti_openpose
    import utils.uti_skeletons_io as uti_skeletons_io
    import utils.uti_commons as uti_commons
    import utils.uti_filter as uti_filter
    import utils.uti_pre_processing as uti_pre_processing


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["s3_pre_processing.py"]

    # common settings

    CLASSES = np.array(config_all["classes"])
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    SKELETON_FILE_NAME_FORMAT = config_all["SKELETON_FILE_NAME_FORMAT"]
    CLIP_NUM_INDEX = config_all["CLIP_NUM_INDEX"]
    ACTION_CLASS_INEDX = config_all["ACTION_CLASS_INEDX"]
    FEATURE_WINDOW_SIZE = config_all["FEATURE_WINDOW_SIZE"]

    # input

    ALL_DETECTED_SKELETONS = par(config["input"]["ALL_DETECTED_SKELETONS"])

    # output
    
    FEATURES = par(config["output"]["FEATURES"])








# -- Functions
def load_numpy_array(ALL_DETECTED_SKELETONS):
    numpy_array = np.load(ALL_DETECTED_SKELETONS)
    skeletons = numpy_array["arr_0"]
    labels = numpy_array["arr_1"]
    action_class = []
    video_clips = []
    for i in range(len(labels)):
        action_class.append(labels[i][ACTION_CLASS_INEDX])
        video_clips.append(labels[i][CLIP_NUM_INDEX])
    return skeletons, action_class, video_clips

def process_features(X0, Y0, video_indices, classes):
    ''' Process features '''
    # Convert features
    # From: raw feature of individual image.
    # To:   time-serials features calculated from multiple raw features
    #       of multiple adjacent images, including speed, normalized pos, etc.
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = extract_features(
            X0, Y0, video_indices, WINDOW_SIZE, 
            is_adding_noise=True, is_print=True)
        X2, Y2 = extract_features(
            X0, Y0, video_indices, WINDOW_SIZE,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = extract_features(
            X0, Y0, video_indices, WINDOW_SIZE, 
            is_adding_noise=False, is_print=True)
        return X, Y

# -- Main


def main_function():
    ''' 
    Load skeleton data from `skeletons_info.txt`, process data, 
    and then save features and labels to .csv file.
    '''

    # Load data
    skeletons, action_class, clip_number = load_numpy_array(ALL_DETECTED_SKELETONS )
    temp_x, temp_y = extract_features(skeletons, action_class, FEATURE_WINDOW_SIZE )
    clips = []
    for i in range(len(labels)):
        clips.append(labels[i][1]) 

    temp_x, temp_y = extract_features(skeletons, action_class, FEATURE_WINDOW_SIZE )

if __name__ == "__main__":
    main_function()
    print("Programms End")

    
__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
