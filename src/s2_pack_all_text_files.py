# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    Get skeletons data from the output of s1_get_skeletons_data.py and put them all in one npz.file
    The ["arr_0"] is for skeletons
        ["arr_1"] is for action classes
    Input:
        DETECTED_SKELETONS_FOLDER
    Output:
        ALL_DETECTED_SKELETONS
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
import cv2
# […]
from os import listdir
from os.path import isfile
from os.path import join
from collections import defaultdict
# Libs
if True:  # Include project path

    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    
    # Own modules
    import utils.uti_images_io as uti_images_io
    import utils.uti_openpose as uti_openpose
    import utils.uti_skeletons_io as uti_skeletons_io
    import utils.uti_commons as uti_commons
    import utils.uti_filter as uti_filter
# […]

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# [Settings] Import the settings from config/config-json file

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["s2_pack_all_text_files.py"]

    # common settings

    CLASSES = np.array(config_all["classes"])
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    SKELETON_FILE_NAME_FORMAT = config_all["SKELETON_FILE_NAME_FORMAT"]
    CLIP_NUM_INDEX = config_all["CLIP_NUM_INDEX"]
    ACTION_CLASS_INEDX = config_all["ACTION_CLASS_INEDX"]
    IMAGES_INFO_INDEX = config_all["IMAGES_INFO_INDEX"]
    # input

    DETECTED_SKELETONS_FOLDER = par(config["input"]["DETECTED_SKELETONS_FOLDER"])

    # output
    
    ALL_DETECTED_SKELETONS = par(config["output"]["ALL_DETECTED_SKELETONS"])
    IMAGES_INFO_SUMMARY = par(config["output"]["IMAGES_INFO_SUMMARY"])
    ALL_SKELETONS_NPY = par(config["output"]["ALL_SKELETONS_NPY"])
    ALL_LABELS_NPY = par(config["output"]["ALL_LABELS_NPY"])
#############################################################################################
def read_all_file_names(sFile_Path, bSort_Lists=True):
    ''' Get all filenames under certain path 
    Arguments:
        sFile_Path {str}: the folder path of input files
    Return:
        skeletons_dir {list}:
            The detected skeletons from s1_get_skeletons_data 
    '''
    sFile_Names = [f for f in listdir(sFile_Path) if isfile(join(sFile_Path, f))]
    if bSort_Lists:
        sFile_Names.sort()
    
    sFile_Names = [sFile_Path + "/" + f for f in sFile_Names]
    return sFile_Names

def read_skeletons_in_single_text(iFile_Number):
    ''' Read the skeletons coordinates from the given text file.
    Arguments:
        iFile_Number {int}: the ith skeleton txt. Each one should only contain two Index, 
                            0 for images infomations and 1 for skeletons
    Return:
        skeletons_dir {list}:
            The detected skeletons from s1_get_skeletons_data 
    '''
    sFile_Names = DETECTED_SKELETONS_FOLDER + \
        SKELETON_FILE_NAME_FORMAT.format(iFile_Number)
    skeletons_dir = uti_commons.read_listlist(sFile_Names)
    skeletons_dir = skeletons_dir[1]
    return skeletons_dir

def read_labels_in_single_text(iFile_Number):
    ''' Read the skeletons coordinates from the given text file.
    Arguments:
        iFile_Number {int}: the ith skeleton txt. Each one should only contain two Index, 
                            0 for images infomations and 1 for skeletons
    Return:
        labels_dir {list}:
            The detected skeletons from s1_get_skeletons_data 
    '''
    sFile_Names = DETECTED_SKELETONS_FOLDER + \
        SKELETON_FILE_NAME_FORMAT.format(iFile_Number)
    labels_dir = uti_commons.read_listlist(sFile_Names)
    labels_dir = labels_dir[IMAGES_INFO_INDEX]
    # labels_dir = labels_dir[3] # name of action class is in[3]
    return labels_dir

def main_function():
    # get all the valid file names and the number
    sFile_Names = read_all_file_names(DETECTED_SKELETONS_FOLDER, bSort_Lists=True)
    iNumber_of_Files = len(sFile_Names)


    # target list to store all the skeletons and labels
    all_skeletons = []
    all_labels = []
    Action_Labels = defaultdict(int)
    # start recording
    for i in range(iNumber_of_Files):
        # get skeletons
        skeletons = read_skeletons_in_single_text(i)
        if not skeletons:
            continue
        all_skeletons.append(skeletons)
        # get action classes
        labels = read_labels_in_single_text(i)
        action_class = labels[ACTION_CLASS_INEDX]
        if action_class not in CLASSES:
            continue
        all_labels.append(labels)
        Action_Labels[action_class] += 1
        print("{}/{}".format(i, iNumber_of_Files))
        # -- Save to npz file
    np.savez(ALL_DETECTED_SKELETONS, ALL_SKELETONS_NPY = all_skeletons, ALL_LABELS_NPY = all_labels)

    # print summary of training images
    images_infos = open(IMAGES_INFO_SUMMARY, 'w')
    line_1 = (f"There are {len(all_skeletons)} skeleton data. \n")
    print(line_1)
    images_infos.write(line_1)
    line_2 = (f"They are saved to {ALL_DETECTED_SKELETONS} \n")
    print(line_2)
    images_infos.write(line_2)
    print("Number of each action: ")
    for labels in CLASSES:
        line_3 = (f"    {labels}: {Action_Labels[labels]} \n")
        images_infos.write(line_3)
        print(line_3)
    images_infos.close()
    print("Programm Ends")
#############################################################################################
# Main function, defaul to read images from web camera
if __name__ == "__main__":
    main_function()












__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
