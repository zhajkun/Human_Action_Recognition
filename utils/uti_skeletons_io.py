# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    This module defines the functions for reading/saving, rebuild and label the skeletons data:

    Functions:
        rebuild_skeletons(skeletons_src)
        cauculate_skeleton_velocity(skeletons_t1, skeletons_t2)
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import numpy as np 
import cv2
import simplejson
# […]

# Own modules
# from {path} import {class}
from operator import abs, sub, add
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    sys.path.append(ROOT + "/home/zhaj/tf-pose-estimation")
    import utils.uti_commons as uti_commons
# […]



# Image info includes: [cnt_action, cnt_clip, cnt_image, img_action_label, filepath]

NaN = 0  # `Not A Number`, which is the value for invalid data.

# -- Functions

def rebuild_skeletons(skeletons_src):
    ''' Rebuild the input skeleton data, change the chain of orders, some joints will appear
    multiple times.More informations please check the document.
    Arguments:
        skeletons_src {list}: contains the joint position of one axis, returned from utils/uti.openpose.humans_to_skels_list()
    Returns:
        skeletons_dir {list}: the skeleton after rebuilded 
    '''

    skeletons_dir = [0]*35  # total numer of joints after rebuild (36 joints)
    skeletons_dirs = []
    skeletons_dir[0] = skeletons_src[0][1]
    skeletons_dir[1] = skeletons_src[0][0]
    skeletons_dir[2] = skeletons_src[0][14]
    skeletons_dir[3] = skeletons_src[0][16]
    skeletons_dir[4] = skeletons_src[0][14]
    skeletons_dir[5] = skeletons_src[0][0]
    skeletons_dir[6] = skeletons_src[0][15]
    skeletons_dir[7] = skeletons_src[0][17]
    skeletons_dir[8] = skeletons_src[0][15]
    skeletons_dir[9] = skeletons_src[0][0]
    skeletons_dir[10] = skeletons_src[0][1]
    skeletons_dir[11] = skeletons_src[0][2]
    skeletons_dir[12] = skeletons_src[0][3]
    skeletons_dir[13] = skeletons_src[0][4]
    skeletons_dir[14] = skeletons_src[0][3]
    skeletons_dir[15] = skeletons_src[0][2]
    skeletons_dir[16] = skeletons_src[0][1]
    skeletons_dir[17] = skeletons_src[0][5]
    skeletons_dir[18] = skeletons_src[0][6]
    skeletons_dir[19] = skeletons_src[0][7]
    skeletons_dir[20] = skeletons_src[0][6]
    skeletons_dir[21] = skeletons_src[0][5]
    skeletons_dir[22] = skeletons_src[0][1]
    skeletons_dir[23] = skeletons_src[0][8]
    skeletons_dir[24] = skeletons_src[0][9]
    skeletons_dir[25] = skeletons_src[0][10]
    skeletons_dir[26] = skeletons_src[0][9]
    skeletons_dir[27] = skeletons_src[0][8]
    skeletons_dir[28] = skeletons_src[0][1]
    skeletons_dir[29] = skeletons_src[0][11]
    skeletons_dir[30] = skeletons_src[0][12]
    skeletons_dir[31] = skeletons_src[0][13]
    skeletons_dir[32] = skeletons_src[0][12]
    skeletons_dir[33] = skeletons_src[0][11]
    skeletons_dir[34] = skeletons_src[0][1]
    skeletons_dirs.append(skeletons_dir)
    return skeletons_dirs

def cauculate_skeleton_velocity(skeletons_t1, skeletons_t2):
    '''Using two skeletons from two continous frames to cauculate the displacements between the them.
    
    Arguments:
        Skeletons_t1 {list}: contains the joint position of one axis, returned from function Rebuild_Skeletons()
        Skeletons_t1 {list}: contains the joint position of one axis, returned from function Rebuild_Skeletons()

    Returns:    
        Velocity {list}: The displacemtns between the two inputs.
    '''
    velocity = list(map(sub, skeletons_t2[0], skeletons_t1[0])) #t2 - t1
    
    return velocity

if __name__ == "__main__":
    s1 = uti_commons.read_listlist('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_DIR/00001.txt')
    s2 = uti_commons.read_listlist('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_DIR/00006.txt')
    
    print(s1)
    print(s2)

    
    res_list = cauculate_skeleton_velocity(s2, s1) 
    print(res_list)
