# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    This Module defines functions for reading images from folder, video, or Webcam
    
    Main classes and functions:

    Read:
        class Read_Images_From_Folder
        class Read_Images_From_Video
        class Read_Images_From_Webcam
    Write:
        class Video_Writer
        class Images_Writer
    
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import time
import multiprocessing
import warnings
import glob
import queue
import threading
import json
import math
# […]

# Libs
import numpy as np 
import cv2
# […]

# Own modules
# from {path} import {class}
# […]
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    ACTION_CLASSES = config_all["ACTION_CLASSES"]
# Main functions

class Read_Images_From_Folder(object):
    ''' Read all images in a given folder, call as module.
    By default, all files under the folder are considered as image file.
    '''

    def __init__(self, sFolder_Path):
        self.sFile_Names = sorted(glob.glob(sFolder_Path + "/*"))
        self._iImages_Counter = 0
        self.sCurrent_File_Name = ""

    def Read_Image(self):
        if self._iImages_Counter >= len(self.sFile_Names):
            return None
        self.sCurrent_File_Name = self.sFile_Names[self._iImages_Counter]
        Image = cv2.imread(self.sCurrent_File_Name, cv2.IMREAD_UNCHANGED)
        self._iImages_Counter += 1
        return Image

    def __len__(self):
        return len(self.sFile_Names)

    def Image_Captured(self):
        return self._iImages_Counter < len(self.sFile_Names)

    def Stop(self):
        None

class Read_Images_From_Video(object):
    def __init__(self, sVideo_Path, iSample_Interval=3):
        ''' Read Images from a video in a given folder, call as module.
        Arguments:
            sVideo_Path {string}: the path of the video folder.
            iSample_Interval {int}: sample every kth image.
        '''
        if not os.path.exists(sVideo_Path):
            raise IOError("Video does not exist: " + sVideo_Path)
        assert isinstance(iSample_Interval, int) and iSample_Interval >= 1
        self._iImages_Counter = 0
        self._bIs_Stoped = False
        self._Video_Captured = cv2.VideoCapture(sVideo_Path)
        Success, Image = self._Video_Captured.read()
        self._Next_Image = Image
        self._iSample_Interval = iSample_Interval
        self._iFPS = self.get_FPS()
        if not self._iFPS >= 0.0001:
            import warnings
            warnings.warn("Invalid fps of video: {}".format(sVideo_Path))

    def Image_Captured(self):
        return self._Next_Image is not None

    def Get_Current_Video_Time(self):
        return 1.0 / self._iFPS * self._iImages_Counter

    def Read_Image(self):
        Image = self._Next_Image
        for i in range(self._iSample_Interval):
            if self._Video_Captured.isOpened():
                Success, Frame = self._Video_Captured.read()
                self._Next_Image = Frame
            else:
                self._Next_Image = None
                break
        self._iImages_Counter += 1
        return Image

    def Stop(self):
        self._Video_Captured.release()
        self._bIs_Stoped = True

    def __del__(self):
        if not self._bIs_Stoped:
            self.Stop()

    def get_FPS(self):

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        # Get video properties
        if int(major_ver) < 3:
            FPS = self._Video_Captured.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            FPS = self._Video_Captured.get(cv2.CAP_PROP_FPS)
        return FPS

class Read_Images_From_Webcam(object):
    def __init__(self, fMax_Framerate=10.0, iWebcam_Index=0):
        ''' Read images from Webcam, call as module.
        Argument:
            fMax_Framerate {float}: the maximum value of the camera framerate.
            iWebcam_Index {int}: index of the web camera. It should be 0 by default.
        '''
        # Settings
        self._fMax_Framerate = fMax_Framerate

        # Initialize video reader
        self._Video = cv2.VideoCapture(iWebcam_Index) # , cv2.CAP_DSHOW)
        self._bIs_Stoped = False

        # Maximal Elements to receive
        iQueue_Size = 3

        # Use a thread to keep on reading images from web camera
        self._Images_Queue = queue.Queue(maxsize=iQueue_Size)
        self._Is_Thread_Alive = multiprocessing.Value('i', 1)
        self._Thread = threading.Thread(
            target=self._Thread_Reading_Webcam_Frames)
        self._Thread.start()

        # Manually control the framerate of the webcam by sleeping
        self._fMin_Duration = 1.0 / self._fMax_Framerate
        self._fPrev_Time = time.time() - 1.0 / fMax_Framerate

    def Read_Image(self):
        fDuration = time.time() - self._fPrev_Time
        if fDuration <= self._fMin_Duration:
            time.sleep(self._fMin_Duration - fDuration)
        self._fPrev_Time = time.time()
        Image = self._Images_Queue.get(timeout=10.0)
        return Image

    def Image_Captured(self):
        return True  # The Webcam always returns a new frame

    def Stop(self):
        self._Is_Thread_Alive.value = False
        self._Video.release()
        self._bIs_Stoped = True

    def __del__(self):
        self.Stop()

    def _Thread_Reading_Webcam_Frames(self):
        while self._Is_Thread_Alive.value:
            Success, Image = self._Video.read()
            if self._Images_Queue.full():  # if queue is full, pop one
                Image_to_Discard = self._Images_Queue.get(timeout=0.001)
            self._Images_Queue.put(Image, timeout=0.001)  # push to queue
        print("Webcam thread is dead.")

class Video_Writer(object):
    def __init__(self, sVideo_Path, fFramerate):
        ''' Read images from web camera, call as module.
        Argument:
            sVideo_Path {string}: The path of the folder.
            fFramerate {intenger}: Frame rate of the recorded video web camera.
        '''

        # -- Settings
        self._sVideo_Path = sVideo_Path
        self._fFramerate = fFramerate

        # -- Variables
        self.__iImages_Counter = 0
        # initialize later when the 1st image comes
        self._video_writer = None
        self._Width = None
        self._Height = None

        # -- Create output folder
        sFolder = os.path.dirname(sVideo_Path)
        if not os.path.exists(sFolder):
            os.makedirs(sFolder)
            sVideo_Path

    def write(self, Image):
        self.__iImages_Counter += 1
        if self.__iImages_Counter == 1:  # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the codec
            self._Width = Image.shape[1]
            self._Height = Image.shape[0]
            self._video_writer = cv2.VideoWriter(
                self._sVideo_Path, fourcc, self._fFramerate, (self._Width, self._Height))
        self._video_writer.write(Image)

    def Stop(self):
        self.__del__()

    def __del__(self):
        if self.__iImages_Counter > 0:
            self._video_writer.release()
            print("Complete writing {}fps and {}s video to {}".format(
                self._fFramerate, self.__iImages_Counter/self._fFramerate, self._sVideo_Path))
       
class Images_Writer(object):
    def __init__(self, sImages_Path, fFramerate):
        ''' Read images from web camera, call as module.
        Argument:
            fMax_Framerate {float}: the real framerate will be reduced below this value.
            iWebcam_Index {int}: index of the web camera. It should be 0 by default.
        '''

        # -- Settings
        self._sVideo_Path = sVideo_Path
        self._fFramerate = fFramerate

        # -- Variables
        self.__iImages_Counter = 0
        # initialize later when the 1st image comes
        self._video_writer = None
        self._Width = None
        self._Height = None

        # -- Create output folder
        sFolder = os.path.dirname(sVideo_Path)
        if not os.path.exists(sFolder):
            os.makedirs(sFolder)
            sVideo_Path

    def Write(self, Image):
        self.__iImages_Counter += 1
        if self.__iImages_Counter == 1:  # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the codec
            self._Width = Image.shape[1]
            self._Height = Image.shape[0]
            self._video_writer = cv2.VideoWriter(
                self._sVideo_Path, fourcc, self._fFramerate, (self._Width, self._Height))
        self._video_writer.Write(Image)

    def Stop(self):
        self.__del__()

    def __del__(self):
        if self.__iImages_Counter > 0:
            self._video_writer.release()
            print("Complete writing {}fps and {}s video to {}".format(
                self._fFramerate, self.__iImages_Counter/self._fFramerate, self._sVideo_Path))

class Image_Displayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._sWindow_Name = "CV2_Display_Window"
        cv2.namedWindow(self._sWindow_Name, cv2.WINDOW_NORMAL)

    def display(self, Image, Wait_Key_ms=1):
        cv2.imshow(self._sWindow_Name, Image)
        cv2.waitKey(Wait_Key_ms)

    def __del__(self):
        cv2.destroyWindow(self._sWindow_Name)

class Image_Displayer_with_Infobox(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._sWindow_Name = "CV2_Display_Window"
        cv2.namedWindow(self._sWindow_Name, cv2.WINDOW_NORMAL)

    def display(self, Image, Wait_Key_ms=1):
        Image_with_border = add_border_to_images(Image)
        cv2.imshow(self._sWindow_Name, Image_with_border)
        cv2.waitKey(Wait_Key_ms)

    def __del__(self):
        cv2.destroyWindow(self._sWindow_Name)

# Functions

def add_white_region_to_left_of_image(img_disp):
    r, c, d = img_disp.shape
    blank = 255 + np.zeros((r, int(c/3), d), np.uint8)
    img_disp = np.hstack((blank, img_disp))
    return img_disp

def add_border_to_images(images_src):
    borderType = cv2.BORDER_CONSTANT
    top = int(0.0 * images_src.shape[0])
    bottom = top
    right = int(0.0 * images_src.shape[1])
    left = int(0.3 * images_src.shape[1])  # shape[1] = cols
    value = [255, 255, 255]
        
    image_dst = cv2.copyMakeBorder(images_src, top, bottom, left, right, borderType, None, value)
    return image_dst

def draw_scores_for_one_person_on_image(images, scores):
    '''
    Draw predicted scores for the first person on images, the first person is the nearest person to image center
    
    '''
    if scores is None:
        return

    for i in range(0, len(ACTION_CLASSES)):

        FONT_SIZE = 0.8
        TXT_X = 20
        TXT_Y = 150 + i*30
        COLOR_INTENSITY = 255


        label = ACTION_CLASSES[i]
        s = "{:<5}: {:.2f}".format(label, scores[i])
        # high light the highes scores
        COLOR_INTENSITY *= (0.0 + 1.0 * scores[i])**0.5

        cv2.putText(images, text=s, org=(TXT_X, TXT_Y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SIZE,
                    color=(0, int(COLOR_INTENSITY), 0), thickness=2)

def draw_bounding_box_for_one_person_on_image(image_src, skeleton_src):
    '''
    '''
    minx = 999
    miny = 999
    maxx = -999
    maxy = -999
    i = 0
    NaN = 0

    while i < len(skeleton_src):
        if not(skeleton_src[i] == NaN or skeleton_src[i+1] == NaN):
            minx = min(minx, skeleton_src[i])
            maxx = max(maxx, skeleton_src[i])
            miny = min(miny, skeleton_src[i+1])
            maxy = max(maxy, skeleton_src[i+1])
        i += 2

    minx = int(minx * image_src.shape[1])
    miny = int(miny * image_src.shape[0])
    maxx = int(maxx * image_src.shape[1])
    maxy = int(maxy * image_src.shape[0])

    # Draw bounding box
    # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
    img_display = cv2.rectangle(
        image_src, (minx, miny), (maxx, maxy), (0, 255, 0), 4)

    # # Draw text at left corner
    # box_scale = max(
    #     0.5, min(2.0, (1.0*(maxx - minx)/image_src.shape[1] / (0.3))**(0.5)))
    # fontsize = 1.4 * box_scale
    # linewidth = int(math.ceil(3 * box_scale))

    # TEST_COL = int(minx + 5 * box_scale)
    # TEST_ROW = int(miny - 10 * box_scale)

    return img_display

def draw_bounding_box_for_multiple_person_on_image(image_src, skeleton_src, scale_h):
    '''
    '''
    if not skeleton_src:
        return 

    for skeleton in skeleton_src:

        skeleton[1::2] = np.divide(skeleton[1::2], scale_h)
        minx = 999
        miny = 999
        maxx = -999
        maxy = -999
        i = 0
        NaN = 0

        while i < len(skeleton):
            if not(skeleton[i] == NaN or skeleton[i+1] == NaN):
                minx = min(minx, skeleton[i])
                maxx = max(maxx, skeleton[i])
                miny = min(miny, skeleton[i+1])
                maxy = max(maxy, skeleton[i+1])
            i += 2

        minx = int(minx * image_src.shape[1])
        miny = int(miny * image_src.shape[0])
        maxx = int(maxx * image_src.shape[1])
        maxy = int(maxy * image_src.shape[0])

        # Draw bounding box
        # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
        img_display = cv2.rectangle(
            image_src, (minx, miny), (maxx, maxy), (0, 255, 0), 1)

        # Draw text at left corner
        box_scale = max(
            0.5, min(2.0, (1.0*(maxx - minx)/image_src.shape[1] / (0.3))**(0.5)))
        fontsize = 1.4 * box_scale
        linewidth = int(math.ceil(3 * box_scale))

        TEST_COL = int(minx + 5 * box_scale)
        TEST_ROW = int(miny - 10 * box_scale)
 
def draw_result_images(image_src, human_ids, skeleton_src, result_dict, scale_h, ACTION_CLASSES):

    font = cv2.FONT_HERSHEY_SIMPLEX

    if not skeleton_src:
        return

    if len(result_dict) >= 1:
        humans_with_score, scores = map(list, zip(*result_dict.items()))

    # draw all skeletons in view
    for idx, skeleton in enumerate(skeleton_src):
        ''' Attention: this variable idx here, is the index of a skeleton in the 
        list of skeletons, it has to be matched with human_ids later (sorted human id) 
       
        '''
        # convert y- axis back    
        skeleton[1::2] = np.divide(skeleton[1::2], scale_h)
        minx = 999
        miny = 999
        maxx = -999
        maxy = -999
        i = 0
        NaN = 0

        while i < len(skeleton):
            if not(skeleton[i] == NaN or skeleton[i+1] == NaN):
                minx = min(minx, skeleton[i])
                maxx = max(maxx, skeleton[i])
                miny = min(miny, skeleton[i+1])
                maxy = max(maxy, skeleton[i+1])
            i += 2

        minx = int(minx * image_src.shape[1])
        miny = int(miny * image_src.shape[0])
        maxx = int(maxx * image_src.shape[1])
        maxy = int(maxy * image_src.shape[0])

        # get the real human id from list[int] human_ids    
        id_in_view = human_ids[idx]

        # Draw bounding box
        # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
        img_display = cv2.rectangle(
            image_src, (minx, miny), (maxx, maxy), (0, 255, 0), 2)

        # Draw text at left corner
        box_scale = max(
            0.5, min(2.0, (1.0*(maxx - minx)/image_src.shape[1] / (0.3))**(0.5)))
        fontsize = 1.2 * box_scale

        linewidth = int(math.ceil(2 * box_scale))

        TEST_COL = int(minx + 5 * box_scale)
        TEST_ROW = int(miny - 10 * box_scale)

        # check if action has been predicted for this human
        if id_in_view in result_dict:
            prediction_matix = result_dict.get(id_in_view)
            sAction_label = convert_label_int_to_str(prediction_matix, ACTION_CLASSES)
        else: 
            sAction_label = ''

        img_display = cv2.putText(
            img_display, "P"+str(id_in_view)+": "+ sAction_label, (TEST_COL, TEST_ROW), font, fontsize, (255, 0, 0), linewidth, cv2.LINE_AA)

def convert_label_int_to_str(prediction_matix, ACTION_CLASSES):

    high_score = max(prediction_matix)
    
    idx = [i for i, j in enumerate(prediction_matix) if j == high_score]
    
    str_label = str(ACTION_CLASSES[idx][0])
    
    return str_label

def save_images(sFile_path, img_src):

    cv2.imwrite(sFile_path, img_src)

def draw_confusion_matrxi_for_training():
    pass
# local test function
def test_Read_From_Webcam():
    ''' Test the class Read_From_Webcam '''
    Webcam_Reader = Read_Images_From_Webcam(fMax_Framerate=10)
    local_Image_Displayer = Image_Displayer()
    import itertools
    for i in itertools.count():
        Image = Webcam_Reader.Read_Image()
        if cv2.waitKey(1) == 27:
            break
        print(f"Read {i}th image...")
        local_Image_Displayer.display(Image)
    Webcam_Reader.Stop()  
    print("Program ends")

def test_Read_From_Video():

    # Get Sources from class Read_From_Video
    sVideo_Path = '/home/zhaj/tf_test/Realtime-Action-Recognition-master/output/01-10-15-16-44-054/'
    iSample_Interval = 1
    Images_From_Video = Read_Images_From_Video(sVideo_Path, iSample_Interval)
    local_Image_Displayer = Image_Displayer()
    import itertools
    for i in itertools.count():
        Image = Read_Images_From_Video.Read_Image(iSample_Interval)
        if Image is None:
            break
        print(f"Read {i}th image...")
        local_Image_Displayer.display(Image)
    print("Program ends")

if __name__ == "__main__":
    test_Read_From_Webcam()

