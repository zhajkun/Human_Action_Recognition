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
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(ROOT)
# Libs
import cv2
import yaml 
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator

# […]
import utils.Test_Utils as Testu
# Own modules

# […]
Testu.fib(2)

print(ROOT)
print(CURR_PATH)
Testu.SimplePrint()









__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
