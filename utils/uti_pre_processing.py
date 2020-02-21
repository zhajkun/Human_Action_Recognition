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
# openpose packages

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
# Own modules
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    FEATURE_WINDOW_SIZE = config_all["FEATURE_WINDOW_SIZE"]
    sys.path.append(ROOT)
  

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

def load_numpy_array(ALL_DETECTED_SKELETONS):
    numpy_array = np.load(ALL_DETECTED_SKELETONS)
    skeletons = numpy_array['ALL_SKELETONS']
    labels = numpy_array['ALL_LABELS']
    # action_class = []
    # video_clips = []
    # for i in range(len(labels)):
    #     action_class.append(labels[i][ACTION_CLASS_INEDX])
    #     video_clips.append(labels[i][CLIP_NUM_INDEX])
    return skeletons, labels

def rebuild_skeleton_joint_order(skeletons_src):
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
    # Start of joints group 1 -- face from R to L
    skeletons_dir[2] = skeletons_src[NOSE_X]
    skeletons_dir[3] = skeletons_src[NOSE_Y]
    skeletons_dir[4] = skeletons_src[R_EYE_X]
    skeletons_dir[5] = skeletons_src[R_EYE_Y]
    skeletons_dir[6] = skeletons_src[R_EAR_X]
    skeletons_dir[7] = skeletons_src[R_EAR_Y]
    skeletons_dir[8] = skeletons_src[R_EYE_X]
    skeletons_dir[9] = skeletons_src[R_EYE_Y]
    skeletons_dir[10] = skeletons_src[NOSE_X]
    skeletons_dir[11] = skeletons_src[NOSE_Y]
    skeletons_dir[12] = skeletons_src[L_EYE_X]
    skeletons_dir[13] = skeletons_src[L_EYE_Y]
    skeletons_dir[14] = skeletons_src[L_EAR_X]
    skeletons_dir[15] = skeletons_src[L_EAR_Y]
    skeletons_dir[16] = skeletons_src[L_EYE_X]
    skeletons_dir[17] = skeletons_src[L_EYE_Y]
    skeletons_dir[18] = skeletons_src[NOSE_X]
    skeletons_dir[19] = skeletons_src[NOSE_Y]
    # End of joints group 1 -- face from R to L
    # Key joint -- Neck
    skeletons_dir[20] = skeletons_src[NECK_X]
    skeletons_dir[21] = skeletons_src[NECK_Y]
    # Start of joints group 2 -- arms from R to L (R part)
    skeletons_dir[22] = skeletons_src[R_SHOULDER_X]
    skeletons_dir[23] = skeletons_src[R_SHOULDER_Y]
    skeletons_dir[24] = skeletons_src[R_ELBOW_X]
    skeletons_dir[25] = skeletons_src[R_ELBOW_Y]
    skeletons_dir[26] = skeletons_src[R_WRIST_X]
    skeletons_dir[27] = skeletons_src[R_WRIST_Y]
    skeletons_dir[28] = skeletons_src[R_ELBOW_X]
    skeletons_dir[29] = skeletons_src[R_ELBOW_Y]
    skeletons_dir[30] = skeletons_src[R_SHOULDER_X]
    skeletons_dir[31] = skeletons_src[R_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (R part)
    # Key joint -- Neck
    skeletons_dir[32] = skeletons_src[NECK_X]
    skeletons_dir[33] = skeletons_src[NECK_Y]
    # Start of joints group 2 -- arms from R to L (L part)
    skeletons_dir[34] = skeletons_src[L_SHOULDER_X]
    skeletons_dir[35] = skeletons_src[L_SHOULDER_Y]
    skeletons_dir[36] = skeletons_src[L_ELBOW_X]
    skeletons_dir[37] = skeletons_src[L_ELBOW_Y]
    skeletons_dir[38] = skeletons_src[L_WRIST_X]
    skeletons_dir[39] = skeletons_src[L_WRIST_Y]
    skeletons_dir[40] = skeletons_src[L_ELBOW_X]
    skeletons_dir[41] = skeletons_src[L_ELBOW_Y]
    skeletons_dir[42] = skeletons_src[L_SHOULDER_X]
    skeletons_dir[43] = skeletons_src[L_SHOULDER_Y]
    # End of joints group 2 -- arms from R to L (L part)
    # Key joint -- Neck
    skeletons_dir[44] = skeletons_src[NECK_X]
    skeletons_dir[45] = skeletons_src[NECK_Y]
    # Start of joints group 3 -- legs from R to L (R part)
    skeletons_dir[46] = skeletons_src[R_HIP_X]
    skeletons_dir[47] = skeletons_src[R_HIP_Y]
    skeletons_dir[48] = skeletons_src[R_KNEE_X]
    skeletons_dir[49] = skeletons_src[R_KNEE_Y]
    skeletons_dir[50] = skeletons_src[R_ANKLE_X]
    skeletons_dir[51] = skeletons_src[R_ANKLE_Y]
    skeletons_dir[52] = skeletons_src[R_KNEE_X]
    skeletons_dir[53] = skeletons_src[R_KNEE_Y]
    skeletons_dir[54] = skeletons_src[R_HIP_X]
    skeletons_dir[55] = skeletons_src[R_HIP_Y]
    # End of joints group 3 -- legs from R to L (R part)
    # Key joint --Neck
    skeletons_dir[56] = skeletons_src[NECK_X]
    skeletons_dir[57] = skeletons_src[NECK_Y]
    # Start of joints group 3 -- legs from R to L (L part)
    skeletons_dir[58] = skeletons_src[L_HIP_X]
    skeletons_dir[59] = skeletons_src[L_HIP_Y]
    skeletons_dir[60] = skeletons_src[L_KNEE_X]
    skeletons_dir[61] = skeletons_src[L_KNEE_Y]
    skeletons_dir[62] = skeletons_src[L_ANKLE_X]
    skeletons_dir[63] = skeletons_src[L_ANKLE_Y]
    skeletons_dir[64] = skeletons_src[L_KNEE_X]
    skeletons_dir[65] = skeletons_src[L_KNEE_Y]
    skeletons_dir[66] = skeletons_src[L_HIP_X]
    skeletons_dir[67] = skeletons_src[L_HIP_Y]
    # End of joints group 3 -- legs from R to L (L part)
    # Key joint --Neck
    skeletons_dir[68] = skeletons_src[NECK_X]
    skeletons_dir[69] = skeletons_src[NECK_Y]
    # End of rebuild
    return skeletons_dir

# def extract_features():

##############################################################################################################

class Features_Generator(object):
    def __init__(self, window_size):
        '''
        Arguments:
            window_size {int}: Number of adjacent frames for extracting features, defined in config/config.json 
        '''
        self._window_size = FEATURE_WINDOW_SIZE
        self.reset()

    def reset(self):
        ''' Reset the Feature_Generator '''
        self._skeletons_deque = deque()
        self._skeletons_prev = None

    def add_cur_skeleton(self, skeleton):
        ''' Input a new skeleton, return the extracted feature.
        Returns:
            is_success {bool}: Return the feature only when
                the historical input skeletons are more than self._window_size.
            features {np.array} 
        '''

        x = retrain_only_body_joints(skeleton)
#######################################################################################################################
    #    if not ProcFtr.has_neck_and_thigh(x):
    #        self.reset()
    #        return False, None

    #    else:
#######################################################################################################################
            # -- Preprocess x
            # Fill zeros, compute angles/lens
            x = self._fill_invalid_data(x)
            if self._is_adding_noise:
                # Add noise druing training stage to augment data
                x = self._add_noises(x, self._noise_intensity)
            x = np.array(x)
            # angles, lens = ProcFtr.joint_pos_2_angle_and_length(x) # deprecate

            # Push to deque
            self._skeletons_deque.append(x)
            # self._angles_deque.append(angles) # deprecate
            # self._lens_deque.append(lens) # deprecate

            self._maintain_deque_size()
            self._skeletons_prev = x.copy()

            # -- Extract features
            if len(self._skeletons_deque) < self._window_size:
                return False, None
            else:
                # -- Normalize all 1~t features
                h_list = [ProcFtr.get_body_height(xi) for xi in self._skeletons_deque]
                mean_height = np.mean(h_list)
                xnorm_list = [ProcFtr.remove_body_offset(xi)/mean_height
                              for xi in self._skeletons_deque]

                # -- Get features of pose/angles/lens
                f_poses = self._deque_features_to_1darray(xnorm_list)
                # f_angles = self._deque_features_to_1darray(self._angles_deque) # deprecate
                # f_lens = self._deque_features_to_1darray(
                #     self._lens_deque) / mean_height # deprecate

                # -- Get features of motion

                f_v_center = self._compute_v_center(
                    self._skeletons_deque, step=1) / mean_height  # len = (t=4)*2 = 8
                f_v_center = np.repeat(f_v_center, 10)  # repeat to add weight

                f_v_joints = self._compute_v_all_joints(
                    xnorm_list, step=1)  # len = (t=(5-1)/step)*12*2 = 96

                # -- Output
                features = np.concatenate((f_poses, f_v_joints, f_v_center))
                return True, features.copy()

    def _maintain_deque_size(self):
        if len(self._skeletons_deque) > self._window_size:
            self._skeletons_deque.popleft()
        if len(self._angles_deque) > self._window_size:
            self._angles_deque.popleft()
        if len(self._lens_deque) > self._window_size:
            self._lens_deque.popleft()

    def _compute_v_center(self, x_deque, step):
        vel = []
        for i in range(0, len(x_deque) - step, step):
            dxdy = x_deque[i+step][0:2] - x_deque[i][0:2]
            vel += dxdy.tolist()
        return np.array(vel)

    def _compute_v_all_joints(self, xnorm_list, step):
        vel = []
        for i in range(0, len(xnorm_list) - step, step):
            dxdy = xnorm_list[i+step][:] - xnorm_list[i][:]
            vel += dxdy.tolist()
        return np.array(vel)

    def _fill_invalid_data(self, x):
        ''' Fill the NaN elements in x with
            their relative-to-neck position in the preious x.
        Argument:
            x {np.array}: a skeleton that has a neck and at least a thigh.
        '''
        res = x.copy()

        def get_px_py_px0_py0(x):
            px = x[0::2]  # list of x
            py = x[1::2]  # list of y
            px0, py0 = get_joint(x, NECK)  # neck
            return px, py, px0, py0
        cur_px, cur_py, cur_px0, cur_py0 = get_px_py_px0_py0(x)
        cur_height = ProcFtr.get_body_height(x)

        is_lack_knee = check_joint(x, L_KNEE) or check_joint(x, R_KNEE)
        is_lack_ankle = check_joint(x, L_ANKLE) or check_joint(x, R_ANKLE)
        if (self._skeletons_prev is None) or is_lack_knee or is_lack_ankle:
            # If preious data is invalid or there is no knee or ankle,
            # then fill the data based on the STAND_SKEL_NORMED.
            for i in range(TOTAL_JOINTS*2):
                if res[i] == NaN:
                    res[i] = (cur_px0 if i % 2 == 0 else cur_py0) + \
                        cur_height * STAND_SKEL_NORMED[i]
            return res

        pre_px, pre_py, pre_px0, pre_py0 = get_px_py_px0_py0(self._skeletons_prev)
        pre_height = ProcFtr.get_body_height(self._skeletons_prev)

        scale = cur_height / pre_height

        bad_idxs = np.nonzero(cur_px == NaN)[0]
        if not len(bad_idxs):  # No invalid data
            return res

        cur_px[bad_idxs] = cur_px0 + (pre_px[bad_idxs] - pre_px0) * scale
        cur_py[bad_idxs] = cur_py0 + (pre_py[bad_idxs] - pre_py0) * scale
        res[::2] = cur_px
        res[1::2] = cur_py
        return res

    def _add_noises(self, x, intensity):
        ''' Add noise to x with a ratio relative to the body height '''
        height = ProcFtr.get_body_height(x)
        randoms = (np.random.random(x.shape, ) - 0.5) * 2 * intensity * height
        x = [(xi + randoms[i] if xi != 0 else xi)
             for i, xi in enumerate(x)]
        return x

    def _deque_features_to_1darray(self, deque_data):
        features = []
        for i in range(len(deque_data)):
            next_feature = deque_data[i].tolist()
            features += next_feature
        features = np.array(features)
        return features

    def _deque_features_to_2darray(self, deque_data):
        features = []
        for i in range(len(deque_data)):
            next_feature = deque_data[i].tolist()
            features.append(next_feature)
        features = np.array(features)
        return features


if __name__ == "__main__":

    skeletons, labels = load_numpy_array('data_proc/Data_Skeletons/all_detected_skeletons.npz')
    test_1 = skeletons[5]

    temp_x = rebuild_skeleton_joint_order(test_1)

    print(temp_x)
__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
