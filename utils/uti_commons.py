# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 

"""
{
    Define some common functions, which will be used in multiple modules.

    Functions:
        save_listlist(sFilepath, ll):
        read_listlist(sFilepath):
        get_time():
        convert_int_to_str(num, blank):
        get_training_images_info(valid_images_list, image_filename_format="{:05d}.jpg"):
    Classes:
        Keyboard_Processor_For_Images_Recorder(object):
        Read_Valid_Images_And_Action_Class(object):
}
{License_info}
"""

# Futures
# […]

# Built-in/Generic Imports
import os
import sys
import math
import time
import glob
# […]

# Libs
import numpy as np # Or any other
import cv2
import simplejson
import datetime
from time import sleep
# […]

# Own modules
# from {path} import {class}
# […]
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

def save_listlist(sFilepath, ll):
    ''' Save a list of lists to file '''
    folder_path = os.path.dirname(sFilepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(sFilepath, 'w') as f:
        simplejson.dump(ll, f)


def read_listlist(sFilepath):
    ''' Read a list of lists from file '''
    with open(sFilepath, 'r') as f:
        ll = simplejson.load(f)

    return ll

# def saveliststofile(sFilepath, ll):
#     folder_path = os.path.dirname(sFilepath)
#     os.makedirs(folder_path, exist_ok=True)
#     with open(sFilepath, 'w') as f:
#         for item in ll:
#             f.write("%s\n" % item)

# def readlists_nosimp(sFilepath):
#     ''' Read a list of lists from file '''
#     with open(sFilepath, 'r') as f:
#        # ll = simplejson.load(f)
#        ll = f.read().splitlines()
#     return ll

def get_time():
    ''' Return the current time on the PC'''
    s=str(datetime.datetime.now())[5:].replace(' ','-').replace(":",'-').replace('.','-')[:-3]
    return s # day, hour, seconds: 02-26-15-51-12-556

def convert_int_to_str(num, blank):
    '''return the input number (which is the number in image counter), and convert it to a blank digital to fit the out put file name format'''
    return ("{:0"+str(blank)+"d}").format(num)

def get_training_images_info(
        valid_images_list,
        image_filename_format="{:05d}.jpg"):
    '''
    Read the descripution file from data/Data_Images/valid_images.txt
    Arguments:
        valid_images_list {str}: path of the txt file that 
            stores the indices and labels of training images.
    Return:
        images_info {list of list}: shape=PxN, where:
            P: number of training images
            N=5: number of tags of that image, including: 
                [iAction_Counter, iClips_Counter, iImages_Counter, sAction_Label, sFilepath]            
    '''
    LEN_IMG_INFO = 5
    images_info = list()

    with open(valid_images_list) as f:

        sFolder_Name = None
        sAction_Label = None
        iAction_Counter = 0
        Actions_Set = set()
        iAction_Images_Counter = dict()
        iClips_Counter = 0
        iImages_Counter = 0

        for iLine_Counter, line in enumerate(f):

            if line.find('_') != -1:  # is True, when it find the first '_', otherwise t will retuen -1
                sFolder_Name = line[:-1]
                sAction_Label = sFolder_Name.split('_')[0]
                if sAction_Label not in Actions_Set:
                    iAction_Counter += 1
                    Actions_Set.add(sAction_Label)
                    iAction_Images_Counter[sAction_Label] = 0

            elif len(line) > 1:  # line != "\n"
                # print("Line {}, len ={}, {}".format(iLine_Counter, len(line), line))
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                iClips_Counter += 1
                for i in range(idx_start, idx_end+1):
                    sFilepath = sFolder_Name+"/" + image_filename_format.format(i)
                    iImages_Counter += 1
                    iAction_Images_Counter[sAction_Label] += 1

                    # Save: 5 values, which are:action class number, clips number, images number, Actions_Set and image file path
                    image_info = [iAction_Counter, iClips_Counter,
                                  iImages_Counter, sAction_Label, sFilepath]
                    assert(len(image_info) == LEN_IMG_INFO)
                    images_info.append(image_info)
                    # An example: [1, 2, 2, 'STANDING', 'STANDING_01-17-16-39-13-104/00065.jpg']

        print("")
        print("Number of action classes = {}".format(len(Actions_Set)))
        print("Number of training images = {}".format(iImages_Counter))
        print("Number of training images of each action:")
        for action in Actions_Set:
            print("  {:>8}| {:>4}|".format(
                action, iAction_Images_Counter[action]))

    return images_info


##############################################################################################################################
# 
#
#
#
#
#
##############################################################################################################################

class Keyboard_Processor_For_Images_Recorder(object):
    ''' Simple class to read keyboad input and to control the programm*'''
    def __init__(self, sub_folder_name, 
                 img_suffix="jpg"):
        self.is_recording = False
        self.path = ROOT + "/"
        self.folder = None
        self.cnt_video = 0
        self.iImages_Counter = 0 
        self.img_suffix = img_suffix
        self.sub_folder_name = sub_folder_name # Default: data/Data_Images/UNDEFINED, change settings via config/config.json

    def check_key_and_save_image(self, q, image):
        
        if q>=0 and chr(q)=='s' and self.is_recording == False:
      
            self.is_recording = True
            self.cnt_video += 1
            self.iImages_Counter = 0
            self.folder = self.sub_folder_name + "_"  + get_time()
       
            if not os.path.exists(self.path + self.folder):
                os.makedirs(self.path + self.folder)

            print("\n\n")
            print("==============================================\n")
            print("Start recording images ...\n")

        if q>=0 and chr(q)=='d' and self.is_recording == True:
            self.is_recording = False
            print("Stop recording images ...\n")
            print("==============================================\n")
            print("\n\n")

        if self.is_recording == True:
            self.iImages_Counter += 1 
            blank = 5
            filename = self.folder + "/" + convert_int_to_str(self.iImages_Counter, blank) + "." + self.img_suffix
            cv2.imwrite(filename, image)
            print("record image: " + filename + "\n")



class Read_Valid_Images_And_Action_Class(object):
    ''' This class will be used to read the training images from /data/Data_Image folder. 
    All the subfolders inside are already recorded by tools/Images_Recorder.py, 
    and named with the action label. The valid images is defined in valid_images.txt 

    '''

    def __init__(self, img_folder, valid_imgs_txt,
                 image_filename_format="{:05d}.jpg"):
        '''
        Arguments:
            img_folder {str}: A folder that contains many sub folders.
                Each subfolder contains many images named as xxxxx.jpg.
            valid_imgs_txt {str}: A txt file which specifies the action labels.
                Example:
                    STANDING_01-17-16-13-57-023
                    58 680

                    WALKING_01-17-16-44-50-670
                    65 393

                    WAVING_01-17-16-49-08-980
                    54 62
                    75 84
            image_filename_format {str}: format of the image filename
        '''
        self.images_info = get_training_images_info(
            valid_imgs_txt, image_filename_format)
        self.imgs_path = img_folder
        self.i = 0
        self.num_images = len(self.images_info)
        print(f"Reading images from txtscript: {img_folder}")
        print(f"Reading images information from: {valid_imgs_txt}")
        print(f"Number of images = {self.num_images}\n")

    def save_images_info(self, sFilepath):
        folder_path = os.path.dirname(sFilepath)
        os.makedirs(folder_path, exist_ok=True)
        with open(sFilepath, 'w') as f:
            simplejson.dump(self.images_info, f)

    def read_image(self):
        '''
        Returns:
            img {RGB image}: 
                Next RGB image from folder. 
            img_action_label {str}: 
                Action label obtained from folder name.
            img_info {list}: 
                Something like [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        Raise:
            RuntimeError, if fail to read next image due to wrong index or wrong sFilepath.
        '''
        self.i += 1
        if self.i > len(self.images_info):
            raise RuntimeError(f"There are only {len(self.images_info)} images, "
                               f"but you try to read the {self.i}th image")
        sFilepath = self.get_filename(self.i)
        img = self.imread(self.i)
        if img is None:
            raise RuntimeError("The image file doesn't exist: " + sFilepath)
        img_action_label = self.get_action_label(self.i)
        img_info = self.get_image_info(self.i)
        return img, img_action_label, img_info

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))

    def get_filename(self, index):
        # The 4th element of
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        # See "get_training_imgs_info" for the data format
        return self.images_info[index-1][4]

    def get_action_label(self, index):
        # The 3rd element of
        # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        # See "get_training_imgs_info" for the data format
        return self.images_info[index-1][3]

    def get_image_info(self, index):
        # Something like [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]
        
        return self.images_info[index-1]
''' Test the functions in this module '''
if __name__ == "__main__":
    file = "data/Data_Images/valid_images.txt"
    s = get_training_images_info(file , image_filename_format="{:05d}.jpg")
    print(s)

