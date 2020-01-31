# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    Get skeletons data from source images, and rebuild the data by change the order of joints.
    This is the 1. Version of this module, which currently only support reading a skeleton of 1 person.
    The source images could be images inside a given folder, a vidoo in a folder or Web_Camera live.
    Read those images and processing it to output the rebuilded
    skeletons coordinates in x- and y- axis seperatly and the diaplacements between them as well
    Input:

    Output:

}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import json
import numpy as np
import cv2
# […]

# Libs
if True:  # Include project path

    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    
    # Own modules
    import utils.uti_images_io as uti_images_io
    import utils.uti_openpose as uti_openpose
    import utils.uti_skeletons_io as uti_skeletons_io
    import utils.uti_commons as uti_commons
# […]

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# [Settings] Import the settings from config/config-jso file

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["s1_get_skeletons_data.py"]

    # common settings

    CLASSES = np.array(config_all["classes"])
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    SKELETON_FILE_NAME_FORMAT = config_all["SKELETON_FILE_NAME_FORMAT"]

    # openpose

    OPENPOSE_MODEL = config["openpose"]["MODEL"]
    OPENPOSE_IMAGE_SIZE = config["openpose"]["IMAGE_SIZE"]

    # input

    SRC_IMAGES_DESCRIPTION_TXT = par(config["input"]["IMAGES_LIST"])
    SRC_IMAGES_FOLDER = par(config["input"]["TRAINING_IMAGES_FOLDER"])

    # output
    
    DST_DETECTED_SKELETONS_FOLDER = par(config["output"]["DETECTED_SKELETONS_FOLDER"])
    DST_IMAGES_WITH_DETECTED_SKELETONS = par(config["output"]["IMAGES_WITH_DETECTED_SKELETONS"])

# Main function, defaul to read images from web camera
if __name__ == "__main__":

    # set the skeleton detector. The two inputs are: operation model and image size
    Sekeleton_Detector = uti_openpose.Skeleton_Detector(OPENPOSE_MODEL, OPENPOSE_IMAGE_SIZE)
    
    Images_Loader = uti_commons.Read_Valid_Images_And_Action_Class(
        img_folder = SRC_IMAGES_FOLDER,
        valid_imgs_txt = SRC_IMAGES_DESCRIPTION_TXT,
        image_filename_format = IMAGE_FILE_NAME_FORMAT)

    # Set the images displayer
    Images_Displayer = uti_images_io.Image_Displayer()   

    # Create the folder for output, if the have not been created
    os.makedirs(DST_DETECTED_SKELETONS_FOLDER, exist_ok= True)
    os.makedirs(DST_IMAGES_WITH_DETECTED_SKELETONS, exist_ok= True)

    iTotal_Number_of_Images = Images_Loader.num_images
    for iImages_Counter in range(iTotal_Number_of_Images):
        # Load training images
        Image, sAction_Class, sImage_Info = Images_Loader.read_image()
        # detect humans
        Humans = Sekeleton_Detector.detect(Image)

        # display detected skeletons on images
        Image_DST = Image.copy()
        Sekeleton_Detector.draw(Image_DST, Humans)
        Images_Displayer.display(Image_DST, 1)

        # save skeletons coordinates to txt files and the sImage_Info with it at the begining
        SKELETONS, SCALE_H = Sekeleton_Detector.humans_to_skeletons_list(Humans)

        SKELETONS.insert(0,sImage_Info)

        sFile_Name = SKELETON_FILE_NAME_FORMAT.format(iImages_Counter)
        uti_commons.save_listlist(
            DST_DETECTED_SKELETONS_FOLDER + sFile_Name, SKELETONS)

        sImage_Name = IMAGE_FILE_NAME_FORMAT.format(iImages_Counter)
        cv2.imwrite(DST_IMAGES_WITH_DETECTED_SKELETONS + sImage_Name, Image_DST)

    print("Programm End")












__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
