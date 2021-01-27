# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

'''
{
    first version of two-stream network
}
{License_info}
'''

# Futures

# […]

# Built-in/Generic Imports
import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt

# Own modules
if True:  # Include project path
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+'/'
    sys.path.append(ROOT)
    import utils.uti_commons as uti_commons

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != '/') else path

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all['train_network.py']

    TXT_FILE_PATH = config['output']['TXT_FILE_PATH']
    FIGURE_PATH = config['output']['FIGURE_PATH']

def main_function():
    acc_0 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/1Frame_bce/all_test_acc.txt')
    loss_0 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/1Frame_bce/all_test_loss.txt')

    acc_1 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/5Frame_bce/all_test_acc.txt')
    loss_1 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/5Frame_bce/all_test_loss.txt')

    acc_2 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/10Frame_bce/all_test_acc.txt')
    loss_2 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/10Frame_bce/all_test_loss.txt')

    acc_3 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/15Frame_bce/all_test_acc.txt')
    loss_3 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/15Frame_bce/all_test_loss.txt')

    acc_4 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_bce/all_test_acc.txt')
    loss_4 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_bce/all_test_loss.txt')

    acc_5 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/25Frame_bce/all_test_acc.txt')
    loss_5 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/25Frame_bce/all_test_loss.txt')

    acc_6 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/30Frame_bce/all_test_acc.txt')
    loss_6 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/30Frame_bce/all_test_loss.txt')

    acc_7 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/35Frame_bce/all_test_acc.txt')
    loss_7 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/35Frame_bce/all_test_loss.txt')

    # print(max(acc_0))  
    # print(max(acc_1)) 
    # print(max(acc_2)) 
    # print(max(acc_3)) 
    # print(max(acc_4)) 
    # print(max(acc_5)) 
    # print(max(acc_6)) 
    # print(max(acc_7)) 

    # print(len(acc_0))

    print(sum(acc_0[50:198])/148)
    print(sum(acc_1[50:198])/148)
    print(sum(acc_2[50:198])/148)
    print(sum(acc_3[50:198])/148)
    print(sum(acc_4[50:198])/148)
    print(sum(acc_5[50:198])/148)
    print(sum(acc_6[50:198])/148)
    print(sum(acc_7[50:198])/148)


if __name__ == '__main__':
    main_function()