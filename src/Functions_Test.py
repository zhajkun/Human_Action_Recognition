# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    This module is used to test the funtions, classes from every implemented files.
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
# -- Main

if __name__ == "__main__":
    # with open(ROOT + 'config/config.json') as json_data_file:
    #     cfg_all = json.load(json_data_file)
    #     cfg = cfg_all["Video_Recorder.py"]

    #     CLASSES = np.array(cfg_all["classes"])
    #     SKELETON_FILE_NAME_FORMAT = cfg_all["SKELETON_FILE_NAME_FORMAT"]
    #     print(CLASSES)
    #     print(SKELETON_FILE_NAME_FORMAT)

    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()

        

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




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