# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 

"""
{Description}
{License_info}
"""

# Futures
from __future__ import print_function
# […]

# Built-in/Generic Imports
import os
import sys
# […]
# Libs
import cv2
import yaml 

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(ROOT)

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator

import utils.uti_images_io as uti_images_io
import utils.uti_openpose as uti_openpose
import utils.uti_skeletons_io as uti_skeletons_io
import utils.uti_commons as uti_commons

# […]

# Own modules

# […]




# -- Main
if __name__ == "__main__":









