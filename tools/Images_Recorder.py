# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

'''
{
    This module is for recording images from a web camera, and save the images in a given folder, sorted by action classes.
    The action classes and folder should be defined in config/config.json file.
}
{License_info}
'''

# Futures

# […]

# Built-in/Generic Imports
import os
import sys

import numpy as np
import cv2
import time
import argparse
import json
import logging
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
# from {path} import {class}
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+'/../'
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+'/'
    sys.path.append(ROOT)
    
    import utils.uti_images_io as uti_images_io
    import utils.uti_commons as uti_commons
# […]

# import settings from config/config.json
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all['Images_Recorder.py']

    CLASSES = np.array(config_all['classes'])
    IMAGE_FILE_NAME_FORMAT = config_all['IMAGE_FILE_NAME_FORMAT']
    RECORDED_IMAGES_FOLDER = config['output']['RECORDED_IMAGES_FOLDER']

def argument_parser():
    parser = argparse.ArgumentParser(
        description='Images Recorder: \nPress s to start recording; Press d to stop recording: Press q to quit.')
    parser.add_argument('--webcam_index', required=False, default=0)

    parser.add_argument('--frame_rate', required=False, default=10.0,
                        help='the frame rate of recording, default is 10 FPS')
    
    parser.add_argument('--action_class', required=False, default='UNDEFINED',
                        help='the action class to record, see config/config.json for details, default is "UNDEFINED"')
    
    args = parser.parse_args()
    return args

def main_function():
    args = argument_parser()
    WEBCAM_INDEX = int(args.webcam_index)
    FRAME_RATE = float(args.frame_rate)
    SUB_FOLDER_NAME = (RECORDED_IMAGES_FOLDER + args.action_class) # This makes the folder: data/Data_Videos/action_classes, if not defined, it UNDEFINED

    image_recorder = uti_commons.Keyboard_Processor_For_Images_Recorder(
        sub_folder_name=SUB_FOLDER_NAME,
        img_suffix='jpg')
    
    Data_Source = uti_images_io.Read_Images_From_Webcam(FRAME_RATE, WEBCAM_INDEX)
    print ('\n'
    '\t   Images_Recorder Demo: \n'
    '     -------------------- \n'
    ' ** Press \'s\' to start record images \n'
    ' ** Press \'d\' to stop recording images \n'
    ' ** Press \'q\' to exit the program ')

    while Data_Source.Image_Captured():
        
        image = Data_Source.Read_Image()

        cv2.imshow('Training Image Recorder', image)
        key = cv2.waitKey(1)
        image_recorder.check_key_and_save_image(key, image)
        if(key>=0 and chr(key)=='q'):
            print('Programm Stopped')
            break
    Data_Source.Stop()
    cv2.destroyAllWindows()


if __name__=='__main__':
    main_function()
    
