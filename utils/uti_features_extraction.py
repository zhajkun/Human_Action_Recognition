# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
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
"""

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
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    LOCAL_OPENPOSE = config_all["TF_OPENPOSE_LOCATION"]
    FEATURE_WINDOW_SIZE = config_all["FEATURE_WINDOW_SIZE"]
    JOINTS_NUMBER = config_all["JOINTS_NUMBER"]
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

    skeletons_dir = [0]*35*2  # total numer of joints after rebuild (35 joints)
    
    
    # Key joint -- Neck
    skeletons_dir[0] = skeletons_src[NECK_X]
    skeletons_dir[1] = skeletons_src[NECK_Y]
    # Start of joints group 2 -- arms from R to L (R part)
    skeletons_dir[2] = skeletons_src[R_SHOULDER_X]
    skeletons_dir[3] = skeletons_src[R_SHOULDER_Y]
    skeletons_dir[4] = skeletons_src[R_ELBOW_X]
    skeletons_dir[5] = skeletons_src[R_ELBOW_Y]
    skeletons_dir[6] = skeletons_src[R_WRIST_X]
    skeletons_dir[7] = skeletons_src[R_WRIST_Y]
    skeletons_dir[8] = skeletons_src[R_ELBOW_X]
    skeletons_dir[9] = skeletons_src[R_ELBOW_Y]
    skeletons_dir[10] = skeletons_src[R_SHOULDER_X]
    skeletons_dir[11] = skeletons_src[R_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (R part)
    # Key joint -- Neck
    skeletons_dir[12] = skeletons_src[NECK_X]
    skeletons_dir[13] = skeletons_src[NECK_Y]
    # Start of joints group 2 -- arms from R to L (L part)
    skeletons_dir[14] = skeletons_src[L_SHOULDER_X]
    skeletons_dir[15] = skeletons_src[L_SHOULDER_Y]
    skeletons_dir[16] = skeletons_src[L_ELBOW_X]
    skeletons_dir[17] = skeletons_src[L_ELBOW_Y]
    skeletons_dir[18] = skeletons_src[L_WRIST_X]
    skeletons_dir[19] = skeletons_src[L_WRIST_Y]
    skeletons_dir[20] = skeletons_src[L_ELBOW_X]
    skeletons_dir[21] = skeletons_src[L_ELBOW_Y]
    skeletons_dir[22] = skeletons_src[L_SHOULDER_X]
    skeletons_dir[23] = skeletons_src[L_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (L part)
    # Key joint -- Neck
    skeletons_dir[24] = skeletons_src[NECK_X]
    skeletons_dir[25] = skeletons_src[NECK_Y]
    # Start of joints group 3 -- legs from R to L (R part)
    skeletons_dir[26] = skeletons_src[R_HIP_X]
    skeletons_dir[27] = skeletons_src[R_HIP_Y]
    skeletons_dir[28] = skeletons_src[R_KNEE_X]
    skeletons_dir[29] = skeletons_src[R_KNEE_Y]
    skeletons_dir[30] = skeletons_src[R_ANKLE_X]
    skeletons_dir[31] = skeletons_src[R_ANKLE_Y]
    skeletons_dir[32] = skeletons_src[R_KNEE_X]
    skeletons_dir[33] = skeletons_src[R_KNEE_Y]
    skeletons_dir[34] = skeletons_src[R_HIP_X]
    skeletons_dir[35] = skeletons_src[R_HIP_Y]
    # End of joints group 3 -- legs from R to L (R part)
    # Key joint --Neck
    skeletons_dir[36] = skeletons_src[NECK_X]
    skeletons_dir[37] = skeletons_src[NECK_Y]
    # Start of joints group 3 -- legs from R to L (L part)
    skeletons_dir[38] = skeletons_src[L_HIP_X]
    skeletons_dir[39] = skeletons_src[L_HIP_Y]
    skeletons_dir[40] = skeletons_src[L_KNEE_X]
    skeletons_dir[41] = skeletons_src[L_KNEE_Y]
    skeletons_dir[42] = skeletons_src[L_ANKLE_X]
    skeletons_dir[43] = skeletons_src[L_ANKLE_Y]
    skeletons_dir[44] = skeletons_src[L_KNEE_X]
    skeletons_dir[45] = skeletons_src[L_KNEE_Y]
    skeletons_dir[46] = skeletons_src[L_HIP_X]
    skeletons_dir[47] = skeletons_src[L_HIP_Y]
    # End of joints group 3 -- legs from R to L (L part)
    # Key joint --Neck
    skeletons_dir[48] = skeletons_src[NECK_X]
    skeletons_dir[49] = skeletons_src[NECK_Y]
    # End of rebuild
    return skeletons_dir

##############################################################################################################

class Features_Generator(object):
    def __init__(self, feature_window_size):
        '''
        Arguments:
            feature_window_size {int}: Number of adjacent frames for extracting features, defined in config/config.json 
        '''
        self._window_size = feature_window_size
        self._reset()

    def _reset(self):
        ''' Reset the Feature_Generator '''
        self._skeletons_deque = deque()
        # self._skeletons_prev = None

    def calculate_features(self, skeleton_src):
        ''' Input a new skeleton, return the extracted feature.
        Arguments:
            skeletons_src {list}: The input new skeleton
        Returns:
            bSuccess {bool}: Return the feature only when
                the historical input skeletons are more than self._window_size.
            features {np.array} 
        '''

        skeleton = rebuild_skeleton_joint_order(skeleton_src)

        # Add the filter here if need, already branched to filter_v1

        skeleton = np.array(skeleton)
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
 
    def _calculate_velocity_in_deque(self, positions, step):
        velocity = []
        zeros_end = [0] * (JOINTS_NUMBER * CHANELS)
        for i in range(len(positions) - 1):
            dxdy = positions[i+step][:] - positions[i][:]
            velocity.append(dxdy.tolist())
        velocity.append(zeros_end)
        return np.array(velocity)




if __name__ == "__main__":
    pass

__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
