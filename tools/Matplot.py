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

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Test Metrics', fontsize=28)

    axes[0].set_ylabel('Loss', fontsize=28)
    # axes[0].set_ylim([0,2])
    axes[0].plot(loss_0, label='1Frame', color='b')
    axes[0].plot(loss_1, label='5Frame', color='g')
    axes[0].plot(loss_2, label='10Frame', color='r')
    axes[0].plot(loss_3, label='15Frame', color='c')
    axes[0].plot(loss_4, label='20Frame', color='m')
    axes[0].plot(loss_5, label='25Frame', color='y')
    axes[0].plot(loss_6, label='30Frame', color='k')
    axes[0].plot(loss_7, label='35Frame', color='darkorange')

    plt.legend()
    axes[1].set_ylabel('Accuracy', fontsize=28)
    axes[1].set_xlabel('Epoch', fontsize=28)
    # axes[1].set_ylim([0.5, 1])
    axes[1].plot(acc_0, label='1Frame', color='b')
    axes[1].plot(acc_1, label='5Frame', color='g')
    axes[1].plot(acc_2, label='10Frame', color='r')
    axes[1].plot(acc_3, label='15Frame', color='c')
    axes[1].plot(acc_4, label='20Frame', color='m')
    axes[1].plot(acc_5, label='25Frame', color='y')
    axes[1].plot(acc_6, label='30Frame', color='k')
    axes[1].plot(acc_7, label='35Frame', color='darkorange')
  
    plt.legend()
    plt.show()
    # plt.savefig(FIGURE_PATH + 'all_test_acc.png', dpi=200)

def main_function_2():
    acc_0 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_bce/all_test_acc.txt')
    loss_0 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_bce/all_test_loss.txt')

    acc_1 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_cc/all_test_acc.txt')
    loss_1 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_cc/all_test_loss.txt')

    acc_2 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_mse/all_test_acc.txt')
    loss_2 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_mse/all_test_loss.txt')

    acc_3 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_mae/all_test_acc.txt')
    loss_3 = uti_commons.read_listlist('Human_Action_Recognition/training_infos/20Frame_mae/all_test_loss.txt')

    acc_0 = acc_0[0:99]
    loss_0 = loss_0[0:99]
    acc_1 = acc_1[0:99]
    loss_1 = loss_1[0:99]
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Test Metrics', fontsize=28)

    axes[0].set_ylabel('Loss', fontsize=28)
    # axes[0].set_ylim([0,2])
    axes[0].plot(loss_0, label='BCE', color='b')
    axes[0].plot(loss_1, label='CC', color='k')
    axes[0].plot(loss_2, label='MSE', color='r')
    axes[0].plot(loss_3, label='MAE', color='g')

    plt.legend()
    axes[1].set_ylabel('Accuracy', fontsize=28)
    axes[1].set_xlabel('Epoch', fontsize=28)
    # axes[1].set_ylim([0.5, 1])
    axes[1].plot(acc_0, label='BCE', color='b')
    axes[1].plot(acc_1, label='CC', color='k')
    axes[1].plot(acc_2, label='MSE', color='r')
    axes[1].plot(acc_3, label='MAE', color='g')

  
    plt.legend()
    plt.show()
    # plt.savefig(FIGURE_PATH + 'all_test_acc.png', dpi=200)

if __name__ == '__main__':
    main_function_2()