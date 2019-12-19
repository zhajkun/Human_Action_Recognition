# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{This module is used to test the funtions, classes from every implemented files.}
{License_info}
"""

# Futures
from __future__ import print_function
# […]

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import numpy as np 
import cv2
import simplejson
# […]

# Own modules
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(ROOT)
import utils.uti_images_io as uti_images_io
import utils.uti_openpose as uti_openpose
import utils.uti_skeletons_io as uti_skeletons_io
import utils.uti_commons as uti_commons
# […]       
def Test_Save_Raw_Skeleton_Data_v1():
    '''Try to extract and display the skeleton data and save it in txt files.'''
    fMax_Framerate = 10
    iCamera_Index = 0
    Data_Source = uti_images_io.Read_Images_From_Webcam(fMax_Framerate,iCamera_Index)
    Image_Window = uti_images_io.Image_Displayer()
    Skeleton_Detector = uti_openpose.SkeletonDetector("mobilenet_thin", "432x368")
    import itertools
    for i in itertools.count():
        img = Data_Source.Read_Image()
        iImage_Counter = 1
        if img is None:
            break
        print(f"Read {i}th image...")

        Detected_Human = Skeleton_Detector.detect(img)
        Image_Output = img.copy()
        Skeleton_Detector.draw(Image_Output, Detected_Human)
        Image_Window.display(Image_Output)
        Lists_To_Save = Skeleton_Detector.humans_to_skels_list(Detected_Human)
        
        
        SKELETONS_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Temp_Skeletons/')
        SKELETON_FILENAME_FORMAT = ('{:05d}.txt')
        filename = SKELETON_FILENAME_FORMAT.format(i)
        uti_commons.save_listlist(
            SKELETONS_FOLDER + filename,
            Lists_To_Save)

        
        ''' Split it into x- and y- coordinates
        Skeletons_X = Lists_To_Save[1::2]
        Skeletons_Y = Lists_To_Save[::2]
        f = open('/home/zhaj/tf_test/Human_Action_Recognition/Temp_Skeletons/X.txt', 'w')
        simplejson.dump(Skeletons_X, f)
        f.close()
        f = open('/home/zhaj/tf_test/Human_Action_Recognition/Temp_Skeletons/Y.txt', 'w')
        simplejson.dump(Skeletons_Y, f)
        f.close()
        print(Skeletons_X)
        print(Skeletons_Y)'''
       

        #sTemp_File_Path = ('')
        #uti_commons.save_listlist()
    print("Program ends")
def Simple_Spilt():
    l = [0,1,2,3,4,5,6,7,8,9]
    print(l)
    odd = l[::2]
    print(odd)
    even = l[1::2]
    print(even)

# -- Main
if __name__ == "__main__":
    Test_Save_Raw_Skeleton_Data_v1()