# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    This module is for recording images from a web camera, and save the video with in a folder, sorted by action classes.
    The action classes and folder should be defined in config/config.json file.
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys

import numpy as np
import cv2
import datetime
import multiprocessing
import queue
import threading
import time
import argparse
import json
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
# from {path} import {class}
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    
    import utils.uti_images_io as uti_images_io
    import utils.uti_commons as uti_commons
# […]

# import settings from config/config.json
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["Images_Recorder.py"]

    CLASSES = np.array(config_all["classes"])
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    RECORDED_IMAGES_FOLDER = config["output"]["RECORDED_IMAGES_FOLDER"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Webcam image recorder: \nPress s to record video; Press d to stop recording.")
    parser.add_argument("--webcam_index", required=False,
                        default=0)
    parser.add_argument("--frame_rate", required=False,
                        default=30.0)
    parser.add_argument("--action_class", required=False,
                        default="UNDEFINED")
    args = parser.parse_args()

    idx = args.webcam_index
    if isinstance(idx, str) and idx.isdigit():
        args.webcam_index = int(idx)
        
    return args
    
args = parse_args()

WEBCAM_INDEX = int(args.webcam_index)
FRAME_RATE = float(args.frame_rate)
SUB_FOLDER_NAME = (RECORDED_IMAGES_FOLDER + args.action_class) # This makes the folder: data/Data_Videos/action_classes, if not defined, it UNDEFINED

if __name__=="__main__":

    image_recorder = uti_commons.Keyboard_Processor_For_Images_Recorder(
        sub_folder_name=SUB_FOLDER_NAME,
        img_suffix="jpg")

    Data_Source = uti_images_io.Read_Images_From_Webcam(FRAME_RATE, WEBCAM_INDEX)

    while Data_Source.Image_Captured():
        
        image = Data_Source.Read_Image()

        cv2.imshow("Training Image Recorder", image)
        key = cv2.waitKey(1)
        image_recorder.check_key_and_save_image(key, image)
        if(key>=0 and chr(key)=='q'):
            break
    Data_Source.Stop()
    cv2.destroyAllWindows()



   