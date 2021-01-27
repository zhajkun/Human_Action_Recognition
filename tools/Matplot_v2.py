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

    acc_1 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/1Frame_cc/all_test_acc.txt')
    loss_1 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/1Frame_cc/all_test_loss.txt')

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Test Metrics', fontsize=28)

    axes[0].set_ylabel('Loss', fontsize=28)
    axes[0].plot(loss_0, label='BCE', color='b')
    axes[0].plot(loss_1, label='CC', color='g')


    plt.legend()
    axes[1].set_ylabel('Accuracy', fontsize=28)
    axes[1].set_xlabel('Epoch', fontsize=28)
    axes[1].plot(acc_0, label='BCE', color='b')
    axes[1].plot(acc_1, label='CC', color='g')

    # font = {'family' : 'DejaVu Sans',
  
    #     'size'   : 28}

    # plt.rc('font', **font)
    plt.legend(fontsize=28)
    plt.show()
    # plt.savefig(FIGURE_PATH + 'all_test_acc.png', dpi=200)
          # 'weight' : 'bold',

if __name__ == '__main__':
    main_function()