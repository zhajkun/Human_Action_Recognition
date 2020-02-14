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
import simplejson
import json
import timeit
import math
import functools
import tensorflow as tf
# […]

# Own modules
if True:
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

# -- Main

if __name__ == "__main__":
    # with open(ROOT + 'config/config.json') as json_data_file:
    #     cfg_all = json.load(json_data_file)
    #     cfg = cfg_all["Video_Recorder.py"]

    #     CLASSES = np.array(cfg_all["classes"])
    #     SKELETON_FILE_NAME_FORMAT = cfg_all["SKELETON_FILE_NAME_FORMAT"]
    #     print(CLASSES)
    #     print(SKELETON_FILE_NAME_FORMAT)





    # cap = cv2.VideoCapture(0)

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret==True:
    #         frame = cv2.flip(frame,0)

    #         # write the flipped frame
    #         out.write(frame)

    #         cv2.imshow('frame',frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    # # Release everything if job is finished
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    # image_filename_format = "{:05d}.jpg"
    # LEN_IMG_INFO = 5
    # images_info = list()
    # images_list = "/home/zhaj/tf_test/Human_Action_Recognition/data/Data_Images/valid_images.txt"
    # with open(images_list) as f:
    #     print('ok)')

    #     folder_name = None
    #     action_label = None
    #     cnt_action = 0
    #     actions = set()
    #     action_images_cnt = dict()
    #     cnt_clip = 0
    #     cnt_image = 0

    #     for cnt_line, line in enumerate(f):

    #         if line.find('_') != -1:  # A new video type
    #             folder_name = line[:-1]
    #             # print(foldedef cmp(a,b):

    #             # print(action_label)
    #             if action_label not in actions:
    #                 cnt_action += 1
    #                 actions.add(action_label)
    #                 action_images_cnt[action_label] = 0

    #         elif len(line) > 1:  # line != "\n"
    #             # print("Line {}, len ={}, {}".format(cnt_line, len(line), line))
    #             indices = [int(s) for s in line.split()]
    #             idx_start = indices[0]
    #             idx_end = indices[1]
    #             cnt_clip += 1
    #             for i in range(idx_start, idx_end+1):
    #                 filepath = folder_name+"/" + image_filename_format.format(i)
    #                 cnt_image += 1
    #                 action_images_cnt[action_label] += 1

    #                 # Save: 5 values, which are:action class number, clips number, images number, actions and image file path
    #                 image_info = [cnt_action, cnt_clip,
    #                               cnt_image, action_label, filepath]
    #                 assert(len(image_info) == LEN_IMG_INFO)
    #                 images_info.append(image_info)
    #                 # An example: [1, 2, 2, 'STANDING', 'STANDING_01-17-16-39-13-104/00065.jpg']

    #     print("") 
    #     print("Number of action classes = {}".format(len(actions)))
    #     print("Number of training images = {}".format(cnt_image))
    #     print("Number of training images of each action:")
    #     for action in actions:
    #         print("  {:>8}| {:>4}|".format(
    #             action, action_images_cnt[action]))




#     mnist = tf.keras.datasets.mnist

#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0
#     model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')])

#     model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#     model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5)

# model.evaluate(x_test,  y_test, verbose=2)


    sss = [0.29878048780487804, 0.3546195652173913, 0.29878048780487804, 0.37907608695652173, 0.2896341463414634, 0.375, 0, 0, 0, 0, 0.31097560975609756, 0.37907608695652173, 0.31097560975609756, 0.4157608695652174, 0.3048780487804878, 0.4279891304347826, 0.28353658536585363, 0.4483695652173913, 0.3231707317073171, 0.46467391304347827, 0.29878048780487804, 0.5380434782608695, 0.29878048780487804, 0.4483695652173913, 0.32926829268292684, 0.46467391304347827, 0.3231707317073171, 0.5339673913043479, 0.29878048780487804, 0.35054347826086957, 0.3018292682926829, 0.35054347826086957, 0, 0, 0.3079268292682927, 0.3546195652173913]

    xxx = sss[::2]
    yyy = sss[1::2]
    print(max(yyy))
    print(min(yyy))