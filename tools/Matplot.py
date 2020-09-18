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
    acc_1 = uti_commons.read_listlist(TXT_FILE_PATH + 'all_test_acc.txt')
    loss_1 = uti_commons.read_listlist('training_infos/all_test_loss.txt')

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Test Metrics')

    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].plot(loss_1)

    axes[1].set_ylabel('Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].plot(acc_1)

    print(max(acc_1))
    plt.savefig(FIGURE_PATH + 'all_test_acc.png', dpi=200)
 

if __name__ == '__main__':
    main_function()