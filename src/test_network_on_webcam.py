# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

'''
{
    first version of two-stream network, change the data source if zou neeed, default is webcam.
    If zou want to run from videos or images from folder, change Images_Loader and specific the data path.
        
        
        
    Input:
        MODEL_PATH
        VIDEO_PATH
        IMAGE_PATH
    Output:
        TEST_OUTPUTS
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
    # input
    MODEL_PATH = par(config['input']['MODEL_PATH'])

    VIDEO_PATH = par(config['input']['VIDEO_PATH']) 

    IMAGE_PATH = par(config['input']['IMAGE_PATH']) 

    # IMAGE_PATH = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data_test/EVA_0/'
    # VIDEO_PATH = 'C:/Users/Kun/tf_test/Human_Action_Recognition/NTU/S001/'
    # output

    TEST_OUTPUTS = par(config['output']['TEST_OUTPUTS'])
    # TEST_RESULTS_FOLDER = par(config['output']['TEST_RESULTS_FOLDER'])
    # TEST_IMAGES_FOLDER = par(config['output']['TEST_IMAGES_FOLDER'])

def predict_action_class(human_ids, statu_list, features_p, features_v, network):
    ''' Argument:
        human_ids {list}: tracked humans in view
        statu_list {list}: tells which deque is full and features is calculated for prediction
        features_p {ndarray}: all positions from feature extractor, if the deque is not full, thhis array is 0
        features_v {ndarray}: all velocity from feature extractor, if the deque is not full, thhis array is 0
        nerwotk {tf.keras model}: the trained network for classification

        Returns:
        predition {dict}: predicted action class of possible valid humans, format {humans_id: action_class_label(str)}
    '''    
    prediction = {}

    for idx, statu in enumerate(statu_list):

        if True == statu:
            positions_temp = np.array(features_p[idx], dtype=float)
            velocity_temp = np.array(features_v[idx], dtype=float)
            
            positions_temp = np.expand_dims(positions_temp, axis=0)
            velocity_temp = np.expand_dims(velocity_temp, axis=0)

            up_0 = positions_temp
            up_1 = positions_temp
            down_0 = velocity_temp
            down_1 = velocity_temp
        
            prediction_vector = network.predict([up_0, up_1, down_0, down_1])

            prediction_int = np.ndarray.tolist(prediction_int)
            
            human_id = human_ids[idx]

            prediction_list = prediction_vector[0].tolist()

            prediction.update({human_id:prediction_list})  
            
    return prediction       

def convert_actionlabel_from_int_to_string(prediction, ACTION_CLASSES):
    ''' Argument:
        prediction {list}: the precition from network, the index of the result is 0 
        ACTION_CLASSES {list}: the pre-defined action classes in string    
    '''
    pass

def main_function():

    # initialize the frames counter at -1, so the first incomming frames is 0
    iFrames_Counter = -1

    # initialize the skeleton detector
    skeleton_detector = uti_openpose.Skeleton_Detector(OPENPOSE_MODEL, OPENPOSE_IMAGE_SIZE)

    # load the trained two stream model
    network = tf.keras.models.load_model(MODEL_PATH)

    # select the data source
    # images_loader = uti_images_io.Read_Images_From_Video(VIDEO_PATH_SRC)
    images_loader = uti_images_io.Read_Images_From_Webcam(10, 0)
    # images_loader = uti_images_io.Read_Images_From_Folder(IMAGE_PATH)
    # initialize the skeleton detector   
    Images_Displayer = uti_images_io.Image_Displayer()
    
    # initialize the skeleton detector
    Featurs_Generator = uti_features_extraction.Features_Generator_Multiple(FEATURE_WINDOW_SIZE)


    # initialize Multiperson Tracker
    Local_Tracker = uti_tracker.Tracker()

    # Recorder = uti_images_io.Video_Writer(TEST_OUTPUTS + 'TEST_'  + uti_commons.get_time() + '/video', 10)
    Timestample = uti_commons.get_time()

    TEST_RESULTS_FOLDER = TEST_OUTPUTS + 'TEST_Webcam'  + Timestample + '/scores/'
    TEST_SKELETONS_FOLDER = TEST_OUTPUTS + 'TEST_Webcam'  + Timestample + '/skeletons/'
    TEST_IMAGES_FOLDER = TEST_OUTPUTS + 'TEST_Webcam'  + Timestample + '/images/'

    if not os.path.exists(TEST_IMAGES_FOLDER):
        os.makedirs(TEST_IMAGES_FOLDER)
    #################################################################################################
    # Will always be ture, if the webcam is pluged in
    while images_loader.Image_Captured():

        # iterate the frames counter by 1
        iFrames_Counter += 1

        # grab frames from data source
        image_src = images_loader.Read_Image()
            
        image_display = image_src.copy()

        # get detected human(s) from openpose
        humans = skeleton_detector.detect(image_src)

        # convert human(s) to 2d coordinates in a list(of lists)
        skeletons_lists_src, scale_h = skeleton_detector.humans_to_skeletons_list(humans)
        
        # delete invalid skeletons from lists
        skeletons_lists = uti_tracker.delete_invalid_skeletons_from_lists(skeletons_lists_src)

        sText_Name = SKELETON_FILE_NAME_FORMAT.format(iFrames_Counter)

        sImage_Name  = IMAGE_FILE_NAME_FORMAT.format(iFrames_Counter)

        uti_commons.save_listlist(TEST_SKELETONS_FOLDER + sText_Name, skeletons_lists_src)

        skeleton_detector.draw(image_display, humans)

        # sort and track humans in frames
        skeletons_dict = Local_Tracker.track(skeletons_lists)

        if len(skeletons_dict) >= 1:
            # get human ids and skeletons seperatly
            human_ids, skeletons_tracked_lists = map(list, zip(*skeletons_dict.items()))

            skeletons_tracked_lists = uti_features_extraction.rebuild_skeleton_joint_order(skeletons_tracked_lists)

            # uti_images_io.draw_bounding_box_for_multiple_person_on_image(image_display, skeletons_tracked_lists, scale_h)

            status_list, features_p, features_v = Featurs_Generator.calculate_features_multiple(human_ids, skeletons_tracked_lists)

            result_dict = predict_action_class(human_ids, status_list, features_p, features_v, network)

            if len(result_dict) > 0:

                values_view = result_dict.values()

                value_iterator = iter(values_view)

                first_value = next(value_iterator)

                # result_str = str(result_dict)

                # np.savetxt(TEST_RESULTS_FOLDER + sText_Name, result_dict)
                uti_commons.save_result_dict(TEST_RESULTS_FOLDER + sText_Name, result_dict)
                
                # only draw all the scores of the first prediction on image
                uti_images_io.draw_scores_for_one_person_on_image(image_display, first_value)

            uti_images_io.draw_result_images(image_display, human_ids, skeletons_tracked_lists, result_dict, scale_h, ACTION_CLASSES)

        cv2.imwrite(TEST_IMAGES_FOLDER + sImage_Name, image_display)
        Images_Displayer.display(image_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Remeber to kill the thread, or yyou can't quit this function properly
    images_loader.Stop()
    
    print('Finished')    

if __name__ == '__main__':
    
    main_function()