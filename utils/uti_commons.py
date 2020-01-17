# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 

"""
{
    Define some common functions, which will be used in multiple modules.
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
import yaml
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

def save_listlist(filepath, ll):
    ''' Save a list of lists to file '''
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(ll, f)


def read_listlist(filepath):
    ''' Read a list of lists from file '''
    with open(filepath, 'r') as f:
        ll = simplejson.load(f)

    return ll

# def saveliststofile(filepath, ll):
#     folder_path = os.path.dirname(filepath)
#     os.makedirs(folder_path, exist_ok=True)
#     with open(filepath, 'w') as f:
#         for item in ll:
#             f.write("%s\n" % item)

# def readlists_nosimp(filepath):
#     ''' Read a list of lists from file '''
#     with open(filepath, 'r') as f:
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

##############################################################################################################################
# 
#
#
#
#
#
##############################################################################################################################

class Keyboard_Processor_For_Images_Recorder(object):
    def __init__(self, sub_folder_name, 
                 img_suffix="jpg"):
        self.is_recording = False
        self.path = ROOT + "/"
        self.folder = None
        self.cnt_video = 0
        self.cnt_image = 0 
        self.img_suffix = img_suffix
        self.sub_folder_name = sub_folder_name # Default: data/Data_Images/UNDEFINED, change settings via config/config.json

    def check_key_and_save_image(self, q, image):
        
        if q>=0 and chr(q)=='s' and self.is_recording == False:
            sleep(1)
            self.is_recording = True
            self.cnt_video += 1
            self.cnt_image = 0
            self.folder = self.sub_folder_name + "_"  + get_time()
            sleep(2)
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
            self.cnt_image += 1 
            blank = 5
            filename = self.folder + "/" + convert_int_to_str(self.cnt_image, blank) + "." + self.img_suffix
            cv2.imwrite(filename, image)
            print("record image: " + filename + "\n")

''' Test the functions in this module '''
if __name__ == "__main__":
    s = convert_int_to_str(0,5)
    print(s)

