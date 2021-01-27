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
import tensorflow as tf
import cv2
import argparse
from os import listdir
from os.path import isfile
from os.path import join
from collections import defaultdict
# […]

# Libs

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
# Own modules
if True:  # Include project path
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+'/'
    sys.path.append(ROOT)
    import utils.uti_data_generator as uti_data_generator
    import utils.uti_commons as uti_commons
    import utils.uti_images_io as uti_images_io
    import utils.uti_openpose as uti_openpose
    import utils.uti_features_extraction as uti_features_extraction
    import utils.uti_filter as uti_filter
    import utils.uti_tracker as uti_tracker
# […]

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != '/') else path

# -- Settings

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all['test_network.py']

    # common settings

    ACTION_CLASSES = np.array(config_all['ACTION_CLASSES'])
    IMAGE_FILE_NAME_FORMAT = config_all['IMAGE_FILE_NAME_FORMAT']
    SKELETON_FILE_NAME_FORMAT = config_all['SKELETON_FILE_NAME_FORMAT']
    VIDEO_FILE_NAME_FORMAT = config_all['VIDEO_FILE_NAME_FORMAT']
    IMAGES_INFO_INDEX = config_all['IMAGES_INFO_INDEX']
    FEATURE_WINDOW_SIZE = config_all['FEATURE_WINDOW_SIZE'] 
    JOINTS_NUMBER = config_all['JOINTS_NUMBER']
    CHANELS = config_all['CHANELS']
    OPENPOSE_MODEL = config_all['OPENPOSE_MODEL']
    OPENPOSE_IMAGE_SIZE = config_all['OPENPOSE_IMAGE_SIZE']
    MODEL_PATH = par(config['input']['MODEL_PATH'])
    
def load_test_datasets(feature_path):
    with np.load(feature_path) as data:
        test_position = data['POSITION_TEST']
        test_velocity = data['VELOCITY_TEST']
        test_labels = data['LABEL_TEST']
    return test_position, test_velocity, test_labels

def load_train_datasets(feature_path):
    with np.load(feature_path) as data:
        train_position = data['POSITION_TRAIN']
        train_velocity = data['VELOCITY_TRAIN'] 
        train_labels = data['LABEL_TRAIN']
    return train_position, train_velocity, train_labels

def plot_from_model():
    # FEATURES_TRAIN = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_train_fw30_c.npz'
    FEATURES_TEST = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_test_fw35_c.npz'
    FIGURE_PATH = 'C:/Users/Kun/tf_test/Human_Action_Recognition/Figures/Train/Confusion_Matrix/'
    network = tf.keras.models.load_model(MODEL_PATH)
    # train_position, train_velocity, train_labels = load_train_datasets(FEATURES_TRAIN)
    test_position, test_velocity, test_labels = load_test_datasets(FEATURES_TEST)
    
    predicted_array = np.zeros([5,5], dtype=int)
    
    # [[0]*5,[0]*5,[0]*5,[0]*5,[0]*5]

    labels_to_plot = test_labels

    for idx in range(len(labels_to_plot)):
        positions_temp = np.array(test_position[idx], dtype=float)
        velocity_temp = np.array(test_velocity[idx], dtype=float)
        
        positions_temp = np.expand_dims(positions_temp, axis=0)
        velocity_temp = np.expand_dims(velocity_temp, axis=0)

        up_0 = positions_temp
        up_1 = positions_temp
        down_0 = velocity_temp
        down_1 = velocity_temp

        prediction = network.predict([up_0, up_1, down_0, down_1])
        prediction_int = np.rint(prediction)
        prediction_int = prediction_int.astype(int)    
        if 0 == labels_to_plot[idx]:
            predicted_array[0] = predicted_array[0] + prediction_int
        elif 1 == labels_to_plot[idx]:
            predicted_array[1] = predicted_array[1] + prediction_int
        elif 2 == labels_to_plot[idx]:
            predicted_array[2] = predicted_array[2] + prediction_int
        elif 3 == labels_to_plot[idx]:
            predicted_array[3] = predicted_array[3] + prediction_int
        elif 4 == labels_to_plot[idx]:
            predicted_array[4] = predicted_array[4] + prediction_int
    # predicted_array = np.rint(predicted_array)
    return predicted_array

def plot_confusion_matrix(cm,
                          target_names,
                          sFile_Name_CM,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    font = {'family' : 'DejaVu Sans',

        'size'   : 22}

    plt.rc('font', **font)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(sFile_Name_CM)

def read_all_file_names(sFile_Path, bSort_Lists=True):
    ''' Get all filenames under certain path 
    Arguments:
        sFile_Path {str}: the folder path of input files
    Return:
        skeletons_dir {list}:
            The detected skeletons from s1_get_skeletons_data 
    '''
    sFile_Names = [f for f in listdir(sFile_Path) if isfile(join(sFile_Path, f))]
    if bSort_Lists:
        sFile_Names.sort()
    
    sFile_Names = [sFile_Path + '/' + f for f in sFile_Names]
    return sFile_Names

def read_all_folder_names(sFolder_Path, bSort_Lists=True):
    ''' Get all filenames under certain path 
    Arguments:
        sFile_Path {str}: the folder path of input files
    Return:
        skeletons_dir {list}:
            The detected skeletons from s1_get_skeletons_data 
    '''
    sFolder_Names = [f for f in listdir(sFolder_Path) if os.path.isdir(join(sFolder_Path, f))]
    if bSort_Lists:
        sFolder_Names.sort()
    
    sFolder_Names = [sFolder_Path + '/' + f for f in sFolder_Names]
    return sFolder_Names

def load_scores(DETECTED_SKELETONS_FOLDER):
    with open(DETECTED_SKELETONS_FOLDER,'r') as inf:
        dict_from_file = eval(inf.read())
    return dict_from_file

def generate_cm(soll_path, ist_path):
    
    predicted_array = np.zeros([5,5], dtype=int)

    DETECTED_SKELETONS_FOLDER = ist_path       

    sFile_Names = read_all_file_names(DETECTED_SKELETONS_FOLDER, bSort_Lists=True)
    
    iNumber_of_Files = len(sFile_Names)

    labels_to_plot = uti_commons.read_listlist(soll_path)

    for idx in range(len(sFile_Names)):

        scores_dict = load_scores(sFile_Names[idx])

        file_idx = sFile_Names[idx][-8: -4]

        file_idx = int(file_idx)
        scores = list(scores_dict.values())

        prediction_int = np.rint(scores[0])
        prediction_int = prediction_int.astype(int)  

        if 0 == labels_to_plot[file_idx]:
            predicted_array[0] = predicted_array[0] + prediction_int
        elif 1 == labels_to_plot[file_idx]:
            predicted_array[1] = predicted_array[1] + prediction_int
        elif 2 == labels_to_plot[file_idx]:
            predicted_array[2] = predicted_array[2] + prediction_int
        elif 3 == labels_to_plot[file_idx]:
            predicted_array[3] = predicted_array[3] + prediction_int
        elif 4 == labels_to_plot[file_idx]:
            predicted_array[4] = predicted_array[4] + prediction_int
        elif 9 == labels_to_plot[file_idx]:
            pass
    print(predicted_array)
    return predicted_array

def get_all_scores(folder_list):
    scores_dir = np.zeros([1,5], dtype=int)

    for i in range(len(folder_list)):
        sublist = read_all_folder_names(folder_list[i])
        for j in range(len(sublist)):
            scores_path_src = sublist[j] + '/scores/'

            if not os.path.isdir(scores_path_src):
                pass
            else:
                scores_names = read_all_file_names(scores_path_src)
                for l in range(len(scores_names)):
                    scores = load_scores(scores_names[l])
                    scores = list(scores.values())
                    for n in range(len(scores)):
                        prediction_int = np.rint(scores[n])
                        prediction_int = prediction_int.astype(int)  
                        scores_dir += prediction_int

    return scores_dir  

def find_unmatch(folder_list):
    scores_dir = np.zeros([1,5], dtype=int)
    folders = []
    
    for i in range(len(folder_list)):
        sublist = read_all_folder_names(folder_list[i])
        for j in range(len(sublist)):
            scores_path_src = sublist[j] + '/scores/'

            if not os.path.isdir(scores_path_src):
                pass
            else:
                scores_names = read_all_file_names(scores_path_src)
                temp_int = 0
                for l in range(len(scores_names)):
                    scores = load_scores(scores_names[l])
                    scores = list(scores.values())
                    for n in range(len(scores)):
                        prediction_int = np.rint(scores[n])
                        prediction_int = prediction_int.astype(int)  
                    
                        if prediction_int[2] == 1:
                            temp_int += 1
                        if temp_int > 5:
                            folders.append(scores_path_src)
                            temp_int = 0
                            

    return folders  


if __name__ == '__main__':


    # plot cm for evalutaion on own dataset
    # ist_path_1  = 'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/Evaluation/EVA_0_30_CC/scores/'
    # soll_path_1 = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_test/EVA_0_labels.txt'
    # cm_1 = generate_cm(soll_path_1, ist_path_1)

    # ist_path_2  = 'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/Evaluation/EVA_1_30_CC/scores/'
    # soll_path_2 = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_test/EVA_1_labels.txt'
    # cm_2 = generate_cm(soll_path_2, ist_path_2)


    # cm = cm_1 + cm_2

    # plot_confusion_matrix(cm, normalize=False, target_names = ['Put in Basket', 'Sitting','Standing', 'Walking', 'Waving'], 
    #                     sFile_Name_CM='Confusion_Matrix_30Frame_CC_EVA.png', title="Confusion Matrix")

    
    # plot cm for evalutaion on irobeka
    # cm = np.zeros([3,3], dtype=int)
    # cm[1,0] = 32
    # cm[1,1] = 652
    # cm[2,0] = 135
    # cm[2,1] = 13
    # cm[2,2] = 1624
    # print(cm)
    # plot_confusion_matrix(cm, normalize=False, target_names = ['Standing', 'Walking', 'Waving'], sFile_Name_CM='Confusion_Matrix_iRobeka.png', title="Confusion Matrix" )



    Walking_059_list = ['C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S001/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S002/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S003/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S004/Walking_059/',                       
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S005/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S006/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S007/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S008/Walking_059/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S009/Walking_059/',                       
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S010/Walking_059/']

    Walking_060_list = ['C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S001/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S002/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S003/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S004/Walking_060/',                       
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S005/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S006/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S007/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S008/Walking_060/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S009/Walking_060/',                       
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S010/Walking_060/']

    Waving_023_list = ['C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S001/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S002/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S003/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S004/Waving_023/',                       
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S005/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S006/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S007/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S008/Waving_023/',
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S009/Waving_023/',                       
                        'C:/Users/Kun/tf_test/Human_Action_Recognition/test_outputs/NTU/S010/Waving_023/']

    line3_1 = get_all_scores(Walking_059_list)
    print(f'059:{line3_1}')

    line3_2 = get_all_scores(Walking_060_list)
    print(f'060:{line3_2}')

    line_4 = get_all_scores(Waving_023_list)
    print(f'023:{line_4}')

    cm = np.zeros([5,5], dtype=int)
    cm[3] = line3_1 + line3_2
 
    cm[4] = line_4

    plot_confusion_matrix(cm, normalize=False, target_names = ['Put in Basket', 'Sitting','Standing', 'Walking', 'Waving'], 
                        sFile_Name_CM='Confusion_Matrix_NTU.png', title="Confusion Matrix")

    
    


