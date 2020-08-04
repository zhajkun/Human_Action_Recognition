# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Load the numpy array and calculate the features, then split the datasets and save them 
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

    import utils.uti_features_extraction as uti_features_extraction
    import utils.uti_commons as uti_commons

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["s3_pre_processing.py"]

    # common settings

    CLASSES = config_all["classes"]
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    SKELETON_FILE_NAME_FORMAT = config_all["SKELETON_FILE_NAME_FORMAT"]
    CLIP_NUM_INDEX = config_all["CLIP_NUM_INDEX"]
    ACTION_CLASS_INT_INEDX = config_all["ACTION_CLASS_INT_INEDX"]
    FEATURE_WINDOW_SIZE = config_all["FEATURE_WINDOW_SIZE"]
    TEST_DATA_SCALE = config_all["TEST_DATA_SCALE"]
    # input

    ALL_DETECTED_SKELETONS = par(config["input"]["ALL_DETECTED_SKELETONS"])

    # output
    FEATURES_TRAIN = par(config["output"]["FEATURES_TRAIN"])
    FEATURES_TEST = par(config["output"]["FEATURES_TEST"])
# -- Functions
def load_numpy_array(ALL_DETECTED_SKELETONS):
    ''' Load the datasets from npz file
    '''
    numpy_array = np.load(ALL_DETECTED_SKELETONS)
    skeletons = numpy_array['ALL_SKELETONS']
    labels = numpy_array['ALL_LABELS']
    action_class_int = []
    video_clips = []
    for i in range(len(labels)):
        action_class_int.append(labels[i][ACTION_CLASS_INT_INEDX])
        video_clips.append(labels[i][CLIP_NUM_INDEX])
    action_class_int_ndarray = np.array(action_class_int, dtype='i')
    video_clips_ndarray = np.array(video_clips, dtype='i')
    return skeletons, action_class_int_ndarray, video_clips_ndarray

def convert_action_to_int(action, CLASSES):
    ''' Convert the input action class name into the correspoding index intenger, may not need this function, because already stored the action label 
        as intenger in the first place
        Arguments:
        action {str}: filmed clips action name from text file.
        CLASSES {list}: all pre defined action classes in config/config.json
        Return:
        CLASSES-index {int}: the index of the action
         '''
    if action in CLASSES:
        return CLASSES.index(action)

def extract_features(
            skeletons, labels, clip_number, window_size):
    ''' From image index and raw skeleton positions,
        Extract features of body velocity, joint velocity, and normalized joint positions.
    '''
    positions_temp = []
    velocity_temp = []
    labels_temp = []
    iClipsCounter = len(clip_number)
    debuger_list = []
    # Loop through all data
    for i, _ in enumerate(clip_number):

        # If a new video clip starts, reset the feature generator
        if i == 0 or clip_number[i] != clip_number[i-1]:
            Features_Generator = uti_features_extraction.Features_Generator(window_size)
        
        # Get features
        success, features_x, features_xs = Features_Generator.calculate_features(skeletons[i, :])
        if success:  # True if (data length > 5) and (skeleton has enough joints)
            positions_temp.append(features_x)       
            velocity_temp.append(features_xs)
            labels_temp.append(labels[i])


        # Print
            print(f"{i+1}/{iClipsCounter}", end=", ")
    positions_temp = np.array(positions_temp)
    velocity_temp = np.array(velocity_temp)
    labels_temp = np.array(labels_temp)
    return positions_temp, velocity_temp, labels_temp

def shuffle_dataset(datasets_position, datasets_velocity, labels, test_percentage):
    indices = np.random.permutation(labels.shape[0])
    valid_cnt = int(labels.shape[0] * test_percentage)
    test_idx, training_idx = indices[:valid_cnt], indices[valid_cnt:]
    test_pos, train_pos = datasets_position[test_idx,:], datasets_position[training_idx,:]
    test_labels, train_labels = labels[test_idx], labels[training_idx]
    test_vel, train_vel = datasets_velocity[test_idx,:], datasets_velocity[training_idx]
    return train_pos, train_vel, train_labels, test_pos, test_vel, test_labels
# -- Main


def main_function():
    ''' 
    Load skeleton data from `skeletons_info.txt`, process data, 
    and then save features and labels to .npz file.
    '''

    # Load data
    skeletons, action_class_int, clip_number = load_numpy_array(ALL_DETECTED_SKELETONS )
    action_class_int = action_class_int 
    # Process Features
    print("\nExtracting time-serials features ...")
    position, velocity, labels = extract_features(skeletons, action_class_int, clip_number, FEATURE_WINDOW_SIZE)
    
    print(f"All Points.shape = {position.shape}, All Velocity.shape = {velocity.shape}")

    position_train, velocity_train, label_train, position_test, velocity_test, label_test = shuffle_dataset(position, velocity, labels, TEST_DATA_SCALE)

    print(f"Train Points.shape = {position_train.shape}, Train Velocity.shape = {velocity_train.shape}")
    print(f"Test Points.shape = {position_test.shape}, Test Velocity.shape = {velocity_test.shape}")
    # Save Features to npz file
    np.savez(FEATURES_TRAIN, POSITION_TRAIN = position_train, VELOCITY_TRAIN = velocity_train, LABEL_TRAIN = label_train)

    np.savez(FEATURES_TEST, POSITION_TEST = position_test, VELOCITY_TEST = velocity_test, LABEL_TEST = label_test)

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
