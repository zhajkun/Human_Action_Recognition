# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Simple function to put several np arrays together.
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import numpy as np
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
# from {path} import {class}
# […]
if True:  # Include project path

    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)


def main_function():
    file_0 = np.load('data_proc/Data_Skeletons/all_detected_skeletons.npz')
    file_1 = np.load('data_proc/Data_Skeletons/all_detected_skeletons_0.npz')
    new_npy_0 = np.concatenate((file_0["arr_0"], file_1["arr_0"]))
    new_npy_1 = np.concatenate((file_0["arr_1"], file_1["arr_1"]))
    np.savez('data_proc/Data_Skeletons/skeletons.npz', new_npy_0, new_npy_1)
    print(file_0["arr_0"])

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

