# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Simple plot for detected skeletons
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.uti_commons as uti_commons
# […]
def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple skeletons ploter: inout the number of skeletons file i.e: 00001.")
    parser.add_argument("--file", required=True,
                        default=00000)
    args = parser.parse_args()      
    return args

args = parse_args()
iFile_Number = args.file


if __name__ == "__main__":
    sFile_Path = 'data_proc/Data_Skeletons/DETECTED_SKELETONS_FOLDER/'
    sFile_Name = sFile_Path + iFile_Number + '.txt'
    skeletons_src = uti_commons.read_listlist(sFile_Name)
    del skeletons_src[0]
    iNum_skeletons = len(skeletons_src)
    skeletons_x = []
    skeletons_y = []
    fig = plt.figure()
    for skeleton in skeletons_src:
        skeletons_x += skeleton[::2]
        skeletons_y += skeleton[1::2]
    plt.plot(skeletons_x, skeletons_y, 'rx')
    plt.xlim(0,1)
    plt.ylim(1,0)  
    plt.show() 
    sys.exit()
__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
