# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

'''
{
    This Module defines functions for processing skeletons data with tf-openpose
    Some of the functions are copied from 'tf-openpose-estimation' and modified.
    
    Main classes and functions:
    Functions:
        _set_logger():
        _set_config():
        _iGet_Input_Image_Size_From_String(sImage_Size):
    

    Classes:
        Skeleton_Detector
}
{License_info}
'''

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import numpy as np
import json
from collections import deque
# openpose packages

# Own modules
# Add tf-pose-estimation project
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+'/'
    sys.path.append(ROOT)
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    LOCAL_OPENPOSE = config_all['TF_OPENPOSE_LOCATION']
    FEATURE_WINDOW_SIZE = config_all['FEATURE_WINDOW_SIZE']
    JOINTS_NUMBER = config_all['JOINTS_NUMBER']
    CHANELS = config_all['CHANELS']
    sys.path.append(ROOT)
    sys.path.append(LOCAL_OPENPOSE)




# Own modules
# from {path} import {class}
# […]

# Functions
if True:
    NOSE_X = 0
    NOSE_Y = 1
    NECK_X = 2
    NECK_Y = 3
    R_SHOULDER_X = 4
    R_SHOULDER_Y = 5
    R_ELBOW_X = 6
    R_ELBOW_Y = 7
    R_WRIST_X = 8
    R_WRIST_Y = 9
    L_SHOULDER_X = 10
    L_SHOULDER_Y = 11
    L_ELBOW_X = 12
    L_ELBOW_Y = 13
    L_WRIST_X = 14
    L_WRIST_Y = 15
    R_HIP_X = 16
    R_HIP_Y = 17
    R_KNEE_X = 18
    R_KNEE_Y = 19
    R_ANKLE_X = 20
    R_ANKLE_Y = 21
    L_HIP_X = 22
    L_HIP_Y = 23
    L_KNEE_X = 24
    L_KNEE_Y = 25
    L_ANKLE_X = 26
    L_ANKLE_Y = 27
    R_EYE_X = 28
    R_EYE_Y = 29
    L_EYE_X = 30
    L_EYE_Y = 31
    R_EAR_X = 32
    R_EAR_Y = 33
    L_EAR_X = 34
    L_EAR_Y = 35

def rebuild_skeleton_joint_order(skeletons_src):
    ''' Rebuild the input skeleton data, change the chain of orders, some joints will appear
    multiple times.More informations please check the document.
    Arguments:
        skeletons_src {list of lists}: contains the joint position of 18 joints and 36 coordinates(x- and y-), 
                                returned from utils/uti.openpose.humans_to_skels_list()
    Returns:
        skeletons_dir {list of lists}: the skeleton after rebuilded, contains 35 joints --> 70 points 
    '''
    skeletons_dirs = []
    if not skeletons_src:
         return 
    for skeleton in skeletons_src:
        skeletons_dir = [0]*35*2  # total numer of joints after rebuild (35 joints)
        # if not skeletons_src:
        #     skeletons_src = skeletons_dir
        # skeletons_src = skeletons_src[0]
        # Key joint -- Neck
        skeletons_dir[0] = skeleton[NECK_X]
        skeletons_dir[1] = skeleton[NECK_Y]
        # Start of joints group 1 -- face from R to L
        skeletons_dir[2] = skeleton[NOSE_X]
        skeletons_dir[3] = skeleton[NOSE_Y]
        skeletons_dir[4] = skeleton[R_EYE_X]
        skeletons_dir[5] = skeleton[R_EYE_Y]
        skeletons_dir[6] = skeleton[R_EAR_X]
        skeletons_dir[7] = skeleton[R_EAR_Y]
        skeletons_dir[8] = skeleton[R_EYE_X]
        skeletons_dir[9] = skeleton[R_EYE_Y]
        skeletons_dir[10] = skeleton[NOSE_X]
        skeletons_dir[11] = skeleton[NOSE_Y]
        skeletons_dir[12] = skeleton[L_EYE_X]
        skeletons_dir[13] = skeleton[L_EYE_Y]
        skeletons_dir[14] = skeleton[L_EAR_X]
        skeletons_dir[15] = skeleton[L_EAR_Y]
        skeletons_dir[16] = skeleton[L_EYE_X]
        skeletons_dir[17] = skeleton[L_EYE_Y]
        skeletons_dir[18] = skeleton[NOSE_X]
        skeletons_dir[19] = skeleton[NOSE_Y]
        # End of joints group 1 -- face from R to L
        # Key joint -- Neck
        skeletons_dir[20] = skeleton[NECK_X]
        skeletons_dir[21] = skeleton[NECK_Y]
        # Start of joints group 2 -- arms from R to L (R part)
        skeletons_dir[22] = skeleton[R_SHOULDER_X]
        skeletons_dir[23] = skeleton[R_SHOULDER_Y]
        skeletons_dir[24] = skeleton[R_ELBOW_X]
        skeletons_dir[25] = skeleton[R_ELBOW_Y]
        skeletons_dir[26] = skeleton[R_WRIST_X]
        skeletons_dir[27] = skeleton[R_WRIST_Y]
        skeletons_dir[28] = skeleton[R_ELBOW_X]
        skeletons_dir[29] = skeleton[R_ELBOW_Y]
        skeletons_dir[30] = skeleton[R_SHOULDER_X]
        skeletons_dir[31] = skeleton[R_SHOULDER_Y]
        # End of joints group 2 -- arms from R to L (R part)
        # Key joint -- Neck
        skeletons_dir[32] = skeleton[NECK_X]
        skeletons_dir[33] = skeleton[NECK_Y]
        # Start of joints group 2 -- arms from R to L (L part)
        skeletons_dir[34] = skeleton[L_SHOULDER_X]
        skeletons_dir[35] = skeleton[L_SHOULDER_Y]
        skeletons_dir[36] = skeleton[L_ELBOW_X]
        skeletons_dir[37] = skeleton[L_ELBOW_Y]
        skeletons_dir[38] = skeleton[L_WRIST_X]
        skeletons_dir[39] = skeleton[L_WRIST_Y]
        skeletons_dir[40] = skeleton[L_ELBOW_X]
        skeletons_dir[41] = skeleton[L_ELBOW_Y]
        skeletons_dir[42] = skeleton[L_SHOULDER_X]
        skeletons_dir[43] = skeleton[L_SHOULDER_Y]
        # End of joints group 2 -- arms from R to L (L part)
        # Key joint -- Neck
        skeletons_dir[44] = skeleton[NECK_X]
        skeletons_dir[45] = skeleton[NECK_Y]
        # Start of joints group 3 -- legs from R to L (R part)
        skeletons_dir[46] = skeleton[R_HIP_X]
        skeletons_dir[47] = skeleton[R_HIP_Y]
        skeletons_dir[48] = skeleton[R_KNEE_X]
        skeletons_dir[49] = skeleton[R_KNEE_Y]
        skeletons_dir[50] = skeleton[R_ANKLE_X]
        skeletons_dir[51] = skeleton[R_ANKLE_Y]
        skeletons_dir[52] = skeleton[R_KNEE_X]
        skeletons_dir[53] = skeleton[R_KNEE_Y]
        skeletons_dir[54] = skeleton[R_HIP_X]
        skeletons_dir[55] = skeleton[R_HIP_Y]
        # End of joints group 3 -- legs from R to L (R part)
        # Key joint --Neck
        skeletons_dir[56] = skeleton[NECK_X]
        skeletons_dir[57] = skeleton[NECK_Y]
        # Start of joints group 3 -- legs from R to L (L part)
        skeletons_dir[58] = skeleton[L_HIP_X]
        skeletons_dir[59] = skeleton[L_HIP_Y]
        skeletons_dir[60] = skeleton[L_KNEE_X]
        skeletons_dir[61] = skeleton[L_KNEE_Y]
        skeletons_dir[62] = skeleton[L_ANKLE_X]
        skeletons_dir[63] = skeleton[L_ANKLE_Y]
        skeletons_dir[64] = skeleton[L_KNEE_X]
        skeletons_dir[65] = skeleton[L_KNEE_Y]
        skeletons_dir[66] = skeleton[L_HIP_X]
        skeletons_dir[67] = skeleton[L_HIP_Y]
        # End of joints group 3 -- legs from R to L (L part)
        # Key joint --Neck
        skeletons_dir[68] = skeleton[NECK_X]
        skeletons_dir[69] = skeleton[NECK_Y]
        # End of rebuild
        skeletons_dirs.append(skeletons_dir)

    return skeletons_dirs

def rebuild_skeleton_joint_order_by_training(skeleton):
    ''' Rebuild the input skeleton data, change the chain of orders, some joints will appear
    multiple times.More informations please check the document.
    Arguments:
        skeletons_src {list of lists}: contains the joint position of 18 joints and 36 coordinates(x- and y-), 
                                returned from utils/uti.openpose.humans_to_skels_list()
    Returns:
        skeletons_dir {list of lists}: the skeleton after rebuilded, contains 35 joints --> 70 points 
    '''
    skeletons_dirs = []
    # for skeleton in skeletons_src:
    skeletons_dir = [0]*35*2  # total numer of joints after rebuild (35 joints)
    # if not skeletons_src:
    #     skeletons_src = skeletons_dir
    # skeletons_src = skeletons_src[0]
    # Key joint -- Neck
    skeletons_dir[0] = skeleton[NECK_X]
    skeletons_dir[1] = skeleton[NECK_Y]
    # Start of joints group 1 -- face from R to L
    skeletons_dir[2] = skeleton[NOSE_X]
    skeletons_dir[3] = skeleton[NOSE_Y]
    skeletons_dir[4] = skeleton[R_EYE_X]
    skeletons_dir[5] = skeleton[R_EYE_Y]
    skeletons_dir[6] = skeleton[R_EAR_X]
    skeletons_dir[7] = skeleton[R_EAR_Y]
    skeletons_dir[8] = skeleton[R_EYE_X]
    skeletons_dir[9] = skeleton[R_EYE_Y]
    skeletons_dir[10] = skeleton[NOSE_X]
    skeletons_dir[11] = skeleton[NOSE_Y]
    skeletons_dir[12] = skeleton[L_EYE_X]
    skeletons_dir[13] = skeleton[L_EYE_Y]
    skeletons_dir[14] = skeleton[L_EAR_X]
    skeletons_dir[15] = skeleton[L_EAR_Y]
    skeletons_dir[16] = skeleton[L_EYE_X]
    skeletons_dir[17] = skeleton[L_EYE_Y]
    skeletons_dir[18] = skeleton[NOSE_X]
    skeletons_dir[19] = skeleton[NOSE_Y]
    # End of joints group 1 -- face from R to L
    # Key joint -- Neck
    skeletons_dir[20] = skeleton[NECK_X]
    skeletons_dir[21] = skeleton[NECK_Y]
    # Start of joints group 2 -- arms from R to L (R part)
    skeletons_dir[22] = skeleton[R_SHOULDER_X]
    skeletons_dir[23] = skeleton[R_SHOULDER_Y]
    skeletons_dir[24] = skeleton[R_ELBOW_X]
    skeletons_dir[25] = skeleton[R_ELBOW_Y]
    skeletons_dir[26] = skeleton[R_WRIST_X]
    skeletons_dir[27] = skeleton[R_WRIST_Y]
    skeletons_dir[28] = skeleton[R_ELBOW_X]
    skeletons_dir[29] = skeleton[R_ELBOW_Y]
    skeletons_dir[30] = skeleton[R_SHOULDER_X]
    skeletons_dir[31] = skeleton[R_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (R part)
    # Key joint -- Neck
    skeletons_dir[32] = skeleton[NECK_X]
    skeletons_dir[33] = skeleton[NECK_Y]
    # Start of joints group 2 -- arms from R to L (L part)
    skeletons_dir[34] = skeleton[L_SHOULDER_X]
    skeletons_dir[35] = skeleton[L_SHOULDER_Y]
    skeletons_dir[36] = skeleton[L_ELBOW_X]
    skeletons_dir[37] = skeleton[L_ELBOW_Y]
    skeletons_dir[38] = skeleton[L_WRIST_X]
    skeletons_dir[39] = skeleton[L_WRIST_Y]
    skeletons_dir[40] = skeleton[L_ELBOW_X]
    skeletons_dir[41] = skeleton[L_ELBOW_Y]
    skeletons_dir[42] = skeleton[L_SHOULDER_X]
    skeletons_dir[43] = skeleton[L_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (L part)
    # Key joint -- Neck
    skeletons_dir[44] = skeleton[NECK_X]
    skeletons_dir[45] = skeleton[NECK_Y]
    # Start of joints group 3 -- legs from R to L (R part)
    skeletons_dir[46] = skeleton[R_HIP_X]
    skeletons_dir[47] = skeleton[R_HIP_Y]
    skeletons_dir[48] = skeleton[R_KNEE_X]
    skeletons_dir[49] = skeleton[R_KNEE_Y]
    skeletons_dir[50] = skeleton[R_ANKLE_X]
    skeletons_dir[51] = skeleton[R_ANKLE_Y]
    skeletons_dir[52] = skeleton[R_KNEE_X]
    skeletons_dir[53] = skeleton[R_KNEE_Y]
    skeletons_dir[54] = skeleton[R_HIP_X]
    skeletons_dir[55] = skeleton[R_HIP_Y]
    # End of joints group 3 -- legs from R to L (R part)
    # Key joint --Neck
    skeletons_dir[56] = skeleton[NECK_X]
    skeletons_dir[57] = skeleton[NECK_Y]
    # Start of joints group 3 -- legs from R to L (L part)
    skeletons_dir[58] = skeleton[L_HIP_X]
    skeletons_dir[59] = skeleton[L_HIP_Y]
    skeletons_dir[60] = skeleton[L_KNEE_X]
    skeletons_dir[61] = skeleton[L_KNEE_Y]
    skeletons_dir[62] = skeleton[L_ANKLE_X]
    skeletons_dir[63] = skeleton[L_ANKLE_Y]
    skeletons_dir[64] = skeleton[L_KNEE_X]
    skeletons_dir[65] = skeleton[L_KNEE_Y]
    skeletons_dir[66] = skeleton[L_HIP_X]
    skeletons_dir[67] = skeleton[L_HIP_Y]
    # End of joints group 3 -- legs from R to L (L part)
    # Key joint --Neck
    skeletons_dir[68] = skeleton[NECK_X]
    skeletons_dir[69] = skeleton[NECK_Y]
    # End of rebuild


    return skeletons_dir

def rebuild_skeleton_joint_order_no_head(skeletons_src):
    ''' Rebuild the input skeleton data, change the chain of orders, some joints will appear
    multiple times.More informations please check the document.
    Arguments:
        skeletons_src {list}: contains the joint position of 18 joints and 36 coordinates(x- and y-), 
                                returned from utils/uti.openpose.humans_to_skels_list()
    Returns:
        skeletons_dir {list}: the skeleton after rebuilded, contains 35 joints --> 70 points 
    '''

    skeletons_dirs = []
    if not skeletons_src:
         return []
    for skeleton in skeletons_src:
        skeletons_dir = [0]*25*2  # total numer of joints after rebuild (35 joints)
    
    
        # Key joint -- Neck
        skeletons_dir[0] = skeleton[NECK_X]
        skeletons_dir[1] = skeleton[NECK_Y]
        # Start of joints group 2 -- arms from R to L (R part)
        skeletons_dir[2] = skeleton[R_SHOULDER_X]
        skeletons_dir[3] = skeleton[R_SHOULDER_Y]
        skeletons_dir[4] = skeleton[R_ELBOW_X]
        skeletons_dir[5] = skeleton[R_ELBOW_Y]
        skeletons_dir[6] = skeleton[R_WRIST_X]
        skeletons_dir[7] = skeleton[R_WRIST_Y]
        skeletons_dir[8] = skeleton[R_ELBOW_X]
        skeletons_dir[9] = skeleton[R_ELBOW_Y]
        skeletons_dir[10] = skeleton[R_SHOULDER_X]
        skeletons_dir[11] = skeleton[R_SHOULDER_Y]
        # End of joints group 2 -- arms from R to L (R part)
        # Key joint -- Neck
        skeletons_dir[12] = skeleton[NECK_X]
        skeletons_dir[13] = skeleton[NECK_Y]
        # Start of joints group 2 -- arms from R to L (L part)
        skeletons_dir[14] = skeleton[L_SHOULDER_X]
        skeletons_dir[15] = skeleton[L_SHOULDER_Y]
        skeletons_dir[16] = skeleton[L_ELBOW_X]
        skeletons_dir[17] = skeleton[L_ELBOW_Y]
        skeletons_dir[18] = skeleton[L_WRIST_X]
        skeletons_dir[19] = skeleton[L_WRIST_Y]
        skeletons_dir[20] = skeleton[L_ELBOW_X]
        skeletons_dir[21] = skeleton[L_ELBOW_Y]
        skeletons_dir[22] = skeleton[L_SHOULDER_X]
        skeletons_dir[23] = skeleton[L_SHOULDER_Y]
        # End of joints group 2 -- arms from R to L (L part)
        # Key joint -- Neck
        skeletons_dir[24] = skeleton[NECK_X]
        skeletons_dir[25] = skeleton[NECK_Y]
        # Start of joints group 3 -- legs from R to L (R part)
        skeletons_dir[26] = skeleton[R_HIP_X]
        skeletons_dir[27] = skeleton[R_HIP_Y]
        skeletons_dir[28] = skeleton[R_KNEE_X]
        skeletons_dir[29] = skeleton[R_KNEE_Y]
        skeletons_dir[30] = skeleton[R_ANKLE_X]
        skeletons_dir[31] = skeleton[R_ANKLE_Y]
        skeletons_dir[32] = skeleton[R_KNEE_X]
        skeletons_dir[33] = skeleton[R_KNEE_Y]
        skeletons_dir[34] = skeleton[R_HIP_X]
        skeletons_dir[35] = skeleton[R_HIP_Y]
        # End of joints group 3 -- legs from R to L (R part)
        # Key joint --Neck
        skeletons_dir[36] = skeleton[NECK_X]
        skeletons_dir[37] = skeleton[NECK_Y]
        # Start of joints group 3 -- legs from R to L (L part)
        skeletons_dir[38] = skeleton[L_HIP_X]
        skeletons_dir[39] = skeleton[L_HIP_Y]
        skeletons_dir[40] = skeleton[L_KNEE_X]
        skeletons_dir[41] = skeleton[L_KNEE_Y]
        skeletons_dir[42] = skeleton[L_ANKLE_X]
        skeletons_dir[43] = skeleton[L_ANKLE_Y]
        skeletons_dir[44] = skeleton[L_KNEE_X]
        skeletons_dir[45] = skeleton[L_KNEE_Y]
        skeletons_dir[46] = skeleton[L_HIP_X]
        skeletons_dir[47] = skeleton[L_HIP_Y]
        # End of joints group 3 -- legs from R to L (L part)
        # Key joint --Neck
        skeletons_dir[48] = skeleton[NECK_X]
        skeletons_dir[49] = skeleton[NECK_Y]
        # End of rebuild
        skeletons_dirs.append(skeletons_dir)
    return skeletons_dirs

def rebuild_skeleton_joint_order_no_head_by_training(skeleton):
    ''' Rebuild the input skeleton data, change the chain of orders, some joints will appear
    multiple times.More informations please check the document.
    Arguments:
        skeletons_src {list of lists}: contains the joint position of 18 joints and 36 coordinates(x- and y-), 
                                returned from utils/uti.openpose.humans_to_skels_list()
    Returns:
        skeletons_dir {list of lists}: the skeleton after rebuilded, contains 35 joints --> 70 points 
    '''
    # for skeleton in skeletons_src:
    skeletons_dir = [0]*25*2  # total numer of joints after rebuild (35 joints)
    # if not skeletons_src:
    #     skeletons_src = skeletons_dir
    # skeletons_src = skeletons_src[0]
    # Key joint -- Neck
        # Key joint -- Neck
    skeletons_dir[0] = skeleton[NECK_X]
    skeletons_dir[1] = skeleton[NECK_Y]
    # Start of joints group 2 -- arms from R to L (R part)
    skeletons_dir[2] = skeleton[R_SHOULDER_X]
    skeletons_dir[3] = skeleton[R_SHOULDER_Y]
    skeletons_dir[4] = skeleton[R_ELBOW_X]
    skeletons_dir[5] = skeleton[R_ELBOW_Y]
    skeletons_dir[6] = skeleton[R_WRIST_X]
    skeletons_dir[7] = skeleton[R_WRIST_Y]
    skeletons_dir[8] = skeleton[R_ELBOW_X]
    skeletons_dir[9] = skeleton[R_ELBOW_Y]
    skeletons_dir[10] = skeleton[R_SHOULDER_X]
    skeletons_dir[11] = skeleton[R_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (R part)
    # Key joint -- Neck
    skeletons_dir[12] = skeleton[NECK_X]
    skeletons_dir[13] = skeleton[NECK_Y]
    # Start of joints group 2 -- arms from R to L (L part)
    skeletons_dir[14] = skeleton[L_SHOULDER_X]
    skeletons_dir[15] = skeleton[L_SHOULDER_Y]
    skeletons_dir[16] = skeleton[L_ELBOW_X]
    skeletons_dir[17] = skeleton[L_ELBOW_Y]
    skeletons_dir[18] = skeleton[L_WRIST_X]
    skeletons_dir[19] = skeleton[L_WRIST_Y]
    skeletons_dir[20] = skeleton[L_ELBOW_X]
    skeletons_dir[21] = skeleton[L_ELBOW_Y]
    skeletons_dir[22] = skeleton[L_SHOULDER_X]
    skeletons_dir[23] = skeleton[L_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (L part)
    # Key joint -- Neck
    skeletons_dir[24] = skeleton[NECK_X]
    skeletons_dir[25] = skeleton[NECK_Y]
    # Start of joints group 3 -- legs from R to L (R part)
    skeletons_dir[26] = skeleton[R_HIP_X]
    skeletons_dir[27] = skeleton[R_HIP_Y]
    skeletons_dir[28] = skeleton[R_KNEE_X]
    skeletons_dir[29] = skeleton[R_KNEE_Y]
    skeletons_dir[30] = skeleton[R_ANKLE_X]
    skeletons_dir[31] = skeleton[R_ANKLE_Y]
    skeletons_dir[32] = skeleton[R_KNEE_X]
    skeletons_dir[33] = skeleton[R_KNEE_Y]
    skeletons_dir[34] = skeleton[R_HIP_X]
    skeletons_dir[35] = skeleton[R_HIP_Y]
    # End of joints group 3 -- legs from R to L (R part)
    # Key joint --Neck
    skeletons_dir[36] = skeleton[NECK_X]
    skeletons_dir[37] = skeleton[NECK_Y]
    # Start of joints group 3 -- legs from R to L (L part)
    skeletons_dir[38] = skeleton[L_HIP_X]
    skeletons_dir[39] = skeleton[L_HIP_Y]
    skeletons_dir[40] = skeleton[L_KNEE_X]
    skeletons_dir[41] = skeleton[L_KNEE_Y]
    skeletons_dir[42] = skeleton[L_ANKLE_X]
    skeletons_dir[43] = skeleton[L_ANKLE_Y]
    skeletons_dir[44] = skeleton[L_KNEE_X]
    skeletons_dir[45] = skeleton[L_KNEE_Y]
    skeletons_dir[46] = skeleton[L_HIP_X]
    skeletons_dir[47] = skeleton[L_HIP_Y]
    # End of joints group 3 -- legs from R to L (L part)
    # Key joint --Neck
    skeletons_dir[48] = skeleton[NECK_X]
    skeletons_dir[49] = skeleton[NECK_Y]
    # End of rebuild


    return skeletons_dir

##############################################################################################################

class Features_Generator(object):
    
    def __init__(self, FEATURE_WINDOW_SIZE):
        '''
        Arguments:
            feature_window_size {int}: Number of adjacent frames for extracting features, defined in config/config.json 
        '''
        self._window_size = FEATURE_WINDOW_SIZE
        self._reset()

    def _reset(self):
        ''' Reset the Feature_Generator '''
        self._skeletons_deque = deque()
        # self._velocity_deque = deque()
        self._skeletons_prev = None

    def calculate_features(self, skeleton_src):
        ''' Input a new skeleton, return the extracted feature.
        Arguments:
            skeletons_src {list}: The input new skeleton
        Returns:
            bSuccess {bool}: Return the feature only when
                the historical input skeletons are more than self._window_size.
            features {np.array} 
        '''
        # skeleton = rebuild_skeleton_joint_order_no_head(skeleton_src)
      

        # Add the filter here if need, already branched to filter_v1

        skeleton = np.array(skeleton_src)
        # Push to deque
        self._skeletons_deque.append(skeleton)

        # self._skeletons_prev = skeleton.copy()

        # -- Extract features
        # check the deque first, if length of deque equals the FEATURE_WINDOW_SIZE, then calculate, if not, pass by
        if len(self._skeletons_deque) < self._window_size:
            return False, None, None
        elif len(self._skeletons_deque) == self._window_size:
            # -- Get features of position and velocity
            position_buff = self._skeletons_deque   
            position = np.array(position_buff)
            velocity = self._calculate_velocity_in_deque(
                position, step=1)  # add one 0 line in this function or else where?
            # -- Output
            self._maintain_deque_size()
            position = np.reshape(position, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            velocity = np.reshape(velocity, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            return True, position.copy(), velocity.copy()
        else:
            self._reset()
            return False, None, None
    
    def _maintain_deque_size(self):
        if len(self._skeletons_deque) == self._window_size:
            self._skeletons_deque.popleft()
            # self._velocity_deque.popleft()
 
    def _calculate_velocity_in_deque(self, positions, step):
        velocity = []
        zeros_end = [0] * (JOINTS_NUMBER * CHANELS)
        for i in range(0, len(positions) - 1, step):
            dxdy = positions[i+step][:] - positions[i][:]
            velocity += dxdy.tolist()
        velocity += zeros_end
        return np.array(velocity)

class Features_Generator_Multiple(Features_Generator):
    
    def __init__(self, FEATURE_WINDOW_SIZE):
        '''
        Arguments:
            feature_window_size {int}: Number of adjacent frames for extracting features, defined in config/config.json 
            human_id {int}: the tracked id of a human
        '''
        self._window_size = FEATURE_WINDOW_SIZE
        self._statu_list = [False, False, False, False, False]
        self._features_list = []
        self._reset()
    
    def _resetstatu_list(self):
        self._statu_list = [False, False, False, False, False]

    def _reset(self):
        ''' Reset the Feature_Generator '''
        self._skeletons_deque_0 = deque()
        self._skeletons_deque_1 = deque()
        self._skeletons_deque_2 = deque()
        self._skeletons_deque_3 = deque()
        self._skeletons_deque_4 = deque()
        # self._velocity_deque = deque()
        self._prev_human_ids = []

    def _check_new_human_ids(self, human_ids):
        '''
        Use this function to compare the new input human IDs with previous human IDs,
        and return IDs to delete , which is not in the new list.
        Arguments:
            human_ids {list}: represent the tracked ids to humans in view. [0,1,2,3,4]
        
        '''
        index_in_prev_ids = []
        prev_ids = set(self._prev_human_ids)
        curr_ids = set(human_ids)
        humans_to_delete = list(prev_ids - curr_ids)

        for id in humans_to_delete:

            id = self._prev_human_ids.index(id)

            index_in_prev_ids.append(id) 

        return index_in_prev_ids, humans_to_delete

    def calculate_features_multiple(self, human_ids, skeletons_tracked_lists):

        self._resetstatu_list()
        index_in_prev_ids, humans_to_delete = self._check_new_human_ids(human_ids)
        # print(index_in_prev_ids)

        self._prev_human_ids = human_ids

        # check if there are ids to delete, if yes, reset the deque coreponding the index
        if index_in_prev_ids and humans_to_delete:

            for idx in index_in_prev_ids:
                if 0 == idx:
                    self._skeletons_deque_0 = deque()
                elif 1 == idx:
                    self._skeletons_deque_1 = deque()
                elif 2 == idx:
                    self._skeletons_deque_2 = deque()
                elif 3 == idx:
                    self._skeletons_deque_3 = deque()
                elif 4 == idx:
                    self._skeletons_deque_4 = deque()
        
        # push new skeletons into deque
        if human_ids:
            # push skeletons into deque 0
            if 1 <= len(human_ids):
                skeleton = np.array(skeletons_tracked_lists[0])
                self._skeletons_deque_0.append(skeleton)
            # push skeletons into deque 1
            if 2 <= len(human_ids):
                skeleton = np.array(skeletons_tracked_lists[1])
                self._skeletons_deque_1.append(skeleton)
            # push skeletons into deque 2
            if 3 <= len(human_ids):
                skeleton = np.array(skeletons_tracked_lists[2])
                self._skeletons_deque_2.append(skeleton)
            # push skeletons into deque 3
            if 4 <= len(human_ids):
                skeleton = np.array(skeletons_tracked_lists[3])
                self._skeletons_deque_3.append(skeleton)
            # push skeletons into deque 4
            if 5 <= len(human_ids):
                skeleton = np.array(skeletons_tracked_lists[4])
                self._skeletons_deque_4.append(skeleton)

        # if non of those deques are full, return none (shortcut for no valid features in deque)   
        if (len(self._skeletons_deque_0) < self._window_size and len(self._skeletons_deque_1) < self._window_size and
                len(self._skeletons_deque_2) < self._window_size and len(self._skeletons_deque_3) < self._window_size and
                    len(self._skeletons_deque_4) < self._window_size):
            return self._statu_list, None, None
        else:

            all_pos = np.empty(shape=(5,FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            all_vel = np.empty(shape=(5,FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            
            if len(self._skeletons_deque_0) == self._window_size:
                position_buff_0 = self._skeletons_deque_0   
                position_0 = np.array(position_buff_0)

                velocity_0 = self._calculate_velocity_in_deque(position_0, step=1)  # add one 0 line in this function or else where?
                # -- Output
            
                position_0 = np.reshape(position_0, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
                velocity_0 = np.reshape(velocity_0, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            
                #pop the first position out of deque
                self._skeletons_deque_0.popleft()

                self._statu_list[0] = True

                all_pos[0] = position_0
                all_vel[0] = velocity_0
            
            if len(self._skeletons_deque_1) == self._window_size:
                
                position_buff_1 = self._skeletons_deque_1   
                position_1 = np.array(position_buff_1)

                velocity_1 = self._calculate_velocity_in_deque(position_1, step=1)  # add one 0 line in this function or else where?
                # -- Output
            
                position_1 = np.reshape(position_1, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
                velocity_1 = np.reshape(velocity_1, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            
                #pop the first position out of deque
                self._skeletons_deque_1.popleft()

                self._statu_list[1] = True

                all_pos[1] = position_1
                all_vel[1] = velocity_1

            if len(self._skeletons_deque_2) == self._window_size:
                
                position_buff_2 = self._skeletons_deque_2   
                position_2 = np.array(position_buff_2)

                velocity_2 = self._calculate_velocity_in_deque(position_2, step=1)  # add one 0 line in this function or else where?
                # -- Output
            
                position_2 = np.reshape(position_2, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
                velocity_2 = np.reshape(velocity_2, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            
                #pop the first position out of deque
                self._skeletons_deque_2.popleft()

                self._statu_list[2] = True

                all_pos[2] = position_2
                all_vel[2] = velocity_2

            if len(self._skeletons_deque_3) == self._window_size:
                
                position_buff_3 = self._skeletons_deque_3   
                position_3 = np.array(position_buff_3)

                velocity_3 = self._calculate_velocity_in_deque(position_3, step=1)  # add one 0 line in this function or else where?
                # -- Output
            
                position_3 = np.reshape(position_3, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
                velocity_3 = np.reshape(velocity_3, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            
                #pop the first position out of deque
                self._skeletons_deque_3.popleft()

                self._statu_list[3] = True
                                
                all_pos[3] = position_3
                all_vel[3] = velocity_3

            if len(self._skeletons_deque_4) == self._window_size:
                
                position_buff_4 = self._skeletons_deque_4   
                position_4 = np.array(position_buff_4)

                velocity_4 = self._calculate_velocity_in_deque(position_4, step=1)  # add one 0 line in this function or else where?
                # -- Output
            
                position_4 = np.reshape(position_4, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
                velocity_4 = np.reshape(velocity_4, (FEATURE_WINDOW_SIZE,JOINTS_NUMBER,CHANELS))
            
                #pop the first position out of deque
                self._skeletons_deque_4.popleft()

                self._statu_list[4] = True

                all_pos[4] = position_4
                all_vel[4] = velocity_4

            statu_list = self._statu_list
            
            self._resetstatu_list()    
            
        return statu_list, all_pos.copy(), all_vel.copy()
if __name__ == '__main__':
    # ww = Features_Generator(FEATURE_WINDOW_SIZE =20)
    x = Features_Generator_Multiple(FEATURE_WINDOW_SIZE =5)

    x._reset()

    fake_list = [[0]*70, [0.1]*70, [0.2]*70, [0.3]*70, [0.4]*70]
    human_ids = [1,2,3,4,5]


    fake_list_t1 = [[0]*70, [0.1]*70, [0.5]*70, [0.6]*70, [0.7]*70]
    human_ids_t1 = [1,2,8,9,10]

    fake_list_t2 = [[0]*70, [0.1]*70, [0.5]*70, [0.6]*70, [0.7]*70]
    human_ids_t2 = [1,2,8,9,10]

    a,b,c = x.calculate_features_multiple(human_ids, fake_list)

    a,b,c = x.calculate_features_multiple(human_ids_t1, fake_list_t1)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)
    
    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)


    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    a,b,c = x.calculate_features_multiple(human_ids_t2, fake_list_t2)

    print(a)



    
