# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    This module is used to test the funtions, classes from every implemented files.
    COCO Output Format
    Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
    Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
    Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
    LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16,
    Left Ear – 17, Background – 18
}
{License_info}
"""

# Futures


# […]

# Built-in/Generic Imports
import os
import sys
# […]

# Libs
import numpy as np 
import cv2
# import simplejson
import json
import timeit
import math
import functools
# import tensorflow as tf
import time

# […]

# Own modules
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    
#     import utils.uti_images_io as uti_images_io
#     import utils.uti_openpose as uti_openpose
#     import utils.uti_skeletons_io as uti_skeletons_io
    import utils.uti_commons as uti_commons
# # […]       
def Test_Save_Raw_Skeleton_Data_v1():
    '''Try to extract and display the skeleton data and save it in txt files.'''
    fMax_Framerate = 10
    iCamera_Index = 0
    Data_Source = uti_images_io.Read_Images_From_Webcam(fMax_Framerate,iCamera_Index)
    Image_Window = uti_images_io.Image_Displayer()
    Skeleton_Detector = uti_openpose.Skeleton_Detector("mobilenet_thin", "432x368")
    # Temporal path and format, use config.yaml file to define it later.
    SKELETON_X_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X/')
    SKELETON_Y_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_Y/')
    SKELETON_X_FOLDER_DIR = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_DIR/')
    SKELETON_Y_FOLDER_DIR = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_Y_DIR/')
    SKELETON_X_FOLDER_VEL = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_VEL/')
    SKELETON_Y_FOLDER_VEL = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_Y_VEL/')
    DST_VIZ_IMGS_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Images/Test_Images/')
    DST_IMGS_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Images/Test_Images_ORI/')
    SKELETON_FILENAME_FORMAT = ('{:05d}.txt')
    IMG_FILENAME_FORMAT = ('{:05d}.jpg')
    import itertools
    for i in itertools.count():
        img = Data_Source.Read_Image()
        if img is None:
            break
        print(f"Read {i}th Frame from Video...")

        Detected_Human = Skeleton_Detector.detect(img)
        Image_Output = img.copy()
        Skeleton_Detector.draw(Image_Output, Detected_Human)
        Image_Window.display(Image_Output)
        Skeleton_X, Skeleton_Y, Scale_h = Skeleton_Detector.humans_to_skels_list(Detected_Human)

        


        
        if Skeleton_X and Skeleton_Y: #only save no-empty lists 
            txt_filename = SKELETON_FILENAME_FORMAT.format(i)
            uti_commons.save_listlist(SKELETON_X_FOLDER + txt_filename, Skeleton_X)
            uti_commons.save_listlist(SKELETON_Y_FOLDER + txt_filename, Skeleton_Y)
            Skeleton_X_DIR = uti_skeletons_io.Rebuild_Skeletons(Skeleton_X)
            Skeleton_Y_DIR = uti_skeletons_io.Rebuild_Skeletons(Skeleton_Y)
            uti_commons.save_listlist(SKELETON_X_FOLDER_DIR + txt_filename, Skeleton_X_DIR)
            uti_commons.save_listlist(SKELETON_Y_FOLDER_DIR + txt_filename, Skeleton_Y_DIR)

            print(f"Saved {i}th Skeleton Data from Webcam...")
            jpg_filename = IMG_FILENAME_FORMAT.format(i)
            cv2.imwrite(
            DST_IMGS_FOLDER + jpg_filename, img)
            cv2.imwrite(
            DST_VIZ_IMGS_FOLDER + jpg_filename, Image_Output)
            print(f"Saved {i}th Image with Skeleton Data from Webcam...")


    print("Program ends")


def Test_Save_Raw_Skeleton_Data_v2():
    fMax_Framerate = 10
    iSample_Interval = 1
    sFolder_Path = ('/home/zhaj/tf_test/Realtime-Action-Recognition-master/data_test/exercise.avi')
    Data_Source = uti_images_io.Read_Images_From_Video(sFolder_Path, iSample_Interval)
    Image_Window = uti_images_io.Image_Displayer()
    Skeleton_Detector = uti_openpose.Skeleton_Detector("mobilenet_thin", "432x368")
    import itertools
    for i in itertools.count():
        img = Data_Source.Read_Image()
        if img is None:
            break
        print(f"Read {i}th Frame from Video...")

        Detected_Human = Skeleton_Detector.detect(img)
        Image_Output = img.copy()
        Skeleton_Detector.draw(Image_Output, Detected_Human)
        Image_Window.display(Image_Output)
        Skeleton_X, Skeleton_Y, Scale_h = Skeleton_Detector.humans_to_skels_list(Detected_Human)

                # Temporal path and format, use config.yaml file to define it later.
        SKELETON_X_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Test_Skeleton_X/')
        SKELETON_Y_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Test_Skeleton_Y/')
        SKELETON_X_FOLDER_DIR = ('/home/zhaj/tf_test/Human_Action_Recognition/Test_Skeleton_X_DIR/')
        SKELETON_Y_FOLDER_DIR = ('/home/zhaj/tf_test/Human_Action_Recognition/Test_Skeleton_Y_DIR/')
        DST_VIZ_IMGS_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Test_Image/')
        SKELETON_FILENAME_FORMAT = ('{:05d}.txt')
        IMG_FILENAME_FORMAT = ('{:05d}.jpg')


        
        if Skeleton_X and Skeleton_Y: #only save no-empty lists 
            txt_filename = SKELETON_FILENAME_FORMAT.format(i)
            uti_commons.saveliststofile(SKELETON_X_FOLDER + txt_filename, Skeleton_X)
            uti_commons.saveliststofile(SKELETON_Y_FOLDER + txt_filename, Skeleton_Y)
            print(f"Saved {i}th Skeleton Data from Video...")
            jpg_filename = IMG_FILENAME_FORMAT.format(i)
            cv2.imwrite(
            DST_VIZ_IMGS_FOLDER + jpg_filename, Image_Output)
            print(f"Saved {i}th Image with Skeleton Data from Video...")



def Test_Rebuild():
    skeletons_src = uti_commons.read_listlist('/home/zhaj/tf_test/Human_Action_Recognition/Temp_Skeletons/00000.txt')
    # skeletons_src = np.array(skeletons)
    skeletons_dir = [0]*35 
    skeletons_dir[0] = skeletons_src[0][1]
    skeletons_dir[1] = skeletons_src[0][0]
    skeletons_dir[2] = skeletons_src[0][14]
    skeletons_dir[3] = skeletons_src[0][16]
    skeletons_dir[4] = skeletons_src[0][14]
    skeletons_dir[5] = skeletons_src[0][0]
    skeletons_dir[6] = skeletons_src[0][15]
    skeletons_dir[7] = skeletons_src[0][17]
    skeletons_dir[8] = skeletons_src[0][15]
    skeletons_dir[9] = skeletons_src[0][0]
    skeletons_dir[10] = skeletons_src[0][1]
    skeletons_dir[11] = skeletons_src[0][2]
    skeletons_dir[12] = skeletons_src[0][3]
    skeletons_dir[13] = skeletons_src[0][4]
    skeletons_dir[14] = skeletons_src[0][3]
    skeletons_dir[15] = skeletons_src[0][2]
    skeletons_dir[16] = skeletons_src[0][1]
    skeletons_dir[17] = skeletons_src[0][5]
    skeletons_dir[18] = skeletons_src[0][6]
    skeletons_dir[19] = skeletons_src[0][7]
    skeletons_dir[20] = skeletons_src[0][6]
    skeletons_dir[21] = skeletons_src[0][5]
    skeletons_dir[22] = skeletons_src[0][1]
    skeletons_dir[23] = skeletons_src[0][8]
    skeletons_dir[24] = skeletons_src[0][9]
    skeletons_dir[25] = skeletons_src[0][10]
    skeletons_dir[26] = skeletons_src[0][9]
    skeletons_dir[27] = skeletons_src[0][8]
    skeletons_dir[28] = skeletons_src[0][1]
    skeletons_dir[29] = skeletons_src[0][11]
    skeletons_dir[30] = skeletons_src[0][12]
    skeletons_dir[31] = skeletons_src[0][13]
    skeletons_dir[32] = skeletons_src[0][12]
    skeletons_dir[33] = skeletons_src[0][11]
    skeletons_dir[34] = skeletons_src[0][1]

    print(skeletons_dir)


def Test_Velocity():
    s1 = uti_commons.readlists_nosimp('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_DIR/00000.txt')
    s2 = uti_commons.readlists_nosimp('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_DIR/00001.txt')
    #res = uti_skeletons_io.Cauculate_Skeleton_Velocity(s1,s2)
    print(type(s1))


def get_training_imgs_info(
        valid_images_list,
        image_filename_format="{:05d}.jpg"):
    '''
    Arguments:
        valid_images_list {str}: path of the txt file that 
            stores the indices and labels of training images.
    Return:
        images_info {list of list}: shape=PxN, where:
            P: number of training images
            N=5: number of tags of that image, including: 
                [cnt_action, cnt_clip, cnt_image, action_label, filepath]
                An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.jpg']                
    '''
    images_info = list()

    with open(valid_images_list) as f:

        folder_name = None
        action_label = None
        cnt_action = 0
        actions = set()
        action_images_cnt = dict()
        cnt_clip = 0
        cnt_image = 0

        for cnt_line, line in enumerate(f):

            if line.find('_') != -1:  # A new video type
                folder_name = line[:-1]
                action_label = folder_name.split('_')[0]
                if action_label not in actions:
                    cnt_action += 1
                    actions.add(action_label)
                    action_images_cnt[action_label] = 0

            elif len(line) > 1:  # line != "\n"
                # print("Line {}, len ={}, {}".format(cnt_line, len(line), line))
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/" + image_filename_format.format(i)
                    cnt_image += 1
                    action_images_cnt[action_label] += 1

                    # Save: 5 values, which are:action class number, clips number, images number, actions and image file path
                    image_info = [cnt_action, cnt_clip,
                                  cnt_image, action_label, filepath]
                    assert(len(image_info) == LEN_IMG_INFO)
                    images_info.append(image_info)
                    # An example: [1, 2, 2, 'STANDING', 'STANDING_01-17-16-39-13-104/00065.jpg']

        print("")
        print("Number of action classes = {}".format(len(actions)))
        print("Number of training images = {}".format(cnt_image))
        print("Number of training images of each action:")
        for action in actions:
            print("  {:>8}| {:>4}|".format(
                action, action_images_cnt[action]))

    return images_info

def cmp(a,b):
    str1 = str(a)+str(b)
    str2 = str(b)+str(a)
    return (str1 > str2) - (str1 < str2)

def get_single_data(file_path, data_index):
    with np.load(file_path) as data:
        datasets_position = data['FEATURES_POSITION'][data_index]
        datasets_velocity = data['FEATURES_VELOCITY'][data_index]
        labels = data['FEATURES_LABELS'][data_index]
    return datasets_position, datasets_velocity, labels
# -- Main
def sort_skeletons_by_dist_to_center(skeletons):
    ''' Skeletons are sorted based on the distance
    between neck and image center, from small to large.
    A skeleton near center will be processed first and be given a smaller human id.
    Here the center is defined as (0.5, 0.5), although it's not accurate due to h_scale.
    '''
    def calc_dist(p1, p2): return (
        (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def cost(skeleton):
        x1, y1 = get_neck(skeleton)
        return calc_dist((x1, y1), (0.5, 0.5))  # dist to center
    def get_neck(skeleton):
        x = skeleton[0]
        y = skeleton[1]
        return x,y
    def cmp(a, b): return (a > b)-(a < b)
    def mycmp(sk1, sk2): return cmp(cost(sk1), cost(sk2))
    sorted_skeletons = sorted(skeletons, key=functools.cmp_to_key(mycmp))
    return sorted_skeletons

if __name__ == "__main__":

    img = cv2.imread('data/Data_Images_10FPS/WALKING_02-06-17-29-30-781/00010.jpg')
    img_display = cv2.rectangle(img, (0, 0), (200, 200), (0, 255, 255), 2)
    cv2.namedWindow('CV2_WINDOW', cv2.WINDOW_NORMAL)
    cv2.imshow('CV2_WINDOW', img_display)
    cv2.waitKey(0)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()