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
import cv2
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

import utils.uti_commons as uti_commons
import utils.uti_images_io as uti_images_io


if __name__ == '__main__':
    source_file = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data/Data_Images/WALKTOME_02-06-17-29-30-781'
    target_file_1 = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data/Data_Images_10FPS/WALKTOME_02-06-17-29-30-781/'
    target_file_2 = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data/Data_Images_10FPS/WALKTOME_02-06-17-29-30-782/'
    target_file_3 = 'C:/Users/Kun/tf_test/Human_Action_Recognition/data/Data_Images_10FPS/WALKTOME_02-06-17-29-30-783/'
    # print len([name for name in os.listdir('.') if os.path.isfile(name)])
    IMAGE_FILE_NAME_FORMAT = '{:05d}.jpg'

    os.makedirs(target_file_1, exist_ok= True)
    os.makedirs(target_file_2, exist_ok= True)
    os.makedirs(target_file_3, exist_ok= True)

    Images_Loader = uti_images_io.Read_Images_From_Folder(source_file)
    Images_Displayer = uti_images_io.Image_Displayer()
    images_counter = 1
    sub_im_counter_1 = 1
    sub_im_counter_2 = 1
    sub_im_counter_3 = 1
    while Images_Loader.Image_Captured():
        images_src = Images_Loader.Read_Image()
        images_display = images_src.copy()
        
        Images_Displayer.display(images_display)

        if 0 == (images_counter % 3):
            sImage_Name = IMAGE_FILE_NAME_FORMAT.format(sub_im_counter_1)
            cv2.imwrite(str(target_file_1) + sImage_Name, images_display)
            sub_im_counter_1 += 1
            
        elif 1 == (images_counter % 3):
            sImage_Name = IMAGE_FILE_NAME_FORMAT.format(sub_im_counter_2)
            cv2.imwrite(str(target_file_2) + sImage_Name, images_display)
            sub_im_counter_2 += 1
     
        elif 2 == (images_counter % 3):
            sImage_Name = IMAGE_FILE_NAME_FORMAT.format(sub_im_counter_3)
            cv2.imwrite(str(target_file_3) + sImage_Name, images_display)
            sub_im_counter_3 += 1
        images_counter += 1
