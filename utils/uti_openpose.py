# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    This Module defines functions for processing skeletons data with tf-openpose
    Some of the functions are copied from 'tf-openpose-estimation' and modified.
    
    Main classes and functions:

    Classes:
        class Skeleton_Detector
    Functions:
        def -set
    
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import time
import argparse
import logging
# […]

# Libs
import cv2
# Add tf-pose-estimation project
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.append(ROOT)
sys.path.append(ROOT + "/home/zhaj/tf-pose-estimation")
# openpose packages

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
# Own modules

# -- Settings
MAX_FRACTION_OF_GPU_TO_USE = 0.4
IS_DRAW_FPS = True

# -- Helper functions
def _set_logger():
    logger = logging.getLogger('TfPoseEstimator')
    logger.setLevel(logging.DEBUG)
    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setLevel(logging.DEBUG)
    logging_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logging_stream_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_stream_handler)
    return logger

def _set_config():
    ''' Set the max GPU memory to use '''
    # For tf 1.13.1, The following setting is needed
    import tensorflow as tf
    from tensorflow import keras
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE
    return config

def _iGet_Input_Image_Size_From_String(sImage_Size):
    ''' If input image_size_str is "123x456", then output (123, 456) '''
    iWidth, iHeight = map(int, sImage_Size.split('x'))
    if iWidth % 16 != 0 or iHeight % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(iWidth), int(iHeight)



# -- Main class

class Skeleton_Detector(object):
    # This class is mainly copied from https://github.com/ildoonet/tf-pose-estimation

    def __init__(self, sModel="cmu", sImage_Size="432x368"):
        ''' Arguments:
            sModel {str}: "cmu" or "mobilenet_thin".        
            sImage_size {str}: resize input images before they are processed. 
                Recommends : 432x368, 336x288, 304x240, 656x368, 
        '''
        # -- Check input
        assert(sModel in ["mobilenet_thin", "cmu"])
        self._iW, self._iH = _iGet_Input_Image_Size_From_String(sImage_Size)
        
        # -- Set up openpose model
        self._sModel = sModel
        self._resize_out_ratio = 4.0 # Resize heatmaps before they are post-processed. If image_size is small, this should be large.
        self._config = _set_config()
        self._tf_pose_estimator = TfPoseEstimator(
            get_graph_path(self._sModel), 
            target_size=(self._iW, self._iH),
            tf_config=self._config)
        self._prev_t = time.time()
        self._iImage_Counter = 0
        
        # -- Set logger
        self._logger = _set_logger()
        

    def detect(self, image):
        ''' Detect human skeleton from image.
        Arguments:
            image: RGB image with arbitrary size. It will be resized to (self._w, self._h).
        Returns:
            humans {list of class Human}: 
                `class Human` is defined in 
                "/home/zhaj/tf-pose-estimation/tf_pose/estimator.py"
                
                The variable `humans` is returned by the function
                `TfPoseEstimator.inference` which is defined in
                `/home/zhaj/tf-pose-estimation/tf_pose/estimator.py`.

                I've written a function `self.humans_to_skels_list` to 
                extract the skeleton from this `class Human` and save the coordinate of x- and y- axis sepratly. 
        '''

        self._iImage_Counter += 1
        if self._iImage_Counter == 1:
            self._image_h = image.shape[0]
            self._image_w = image.shape[1]
            self._scale_h = 1.0 * self._image_h / self._image_w
        t = time.time()

        # Do inference
        humans = self._tf_pose_estimator.inference(
            image, resize_to_default=(self._iW > 0 and self._iH > 0),
            upsample_size=self._resize_out_ratio)

        # Print result and time cost
        elapsed = time.time() - t
        self._logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        ''' Draw human skeleton on img_disp inplace.
        Argument:
            img_disp {RGB image}
            humans {a class returned by self.detect}
        '''
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        if IS_DRAW_FPS:
            cv2.putText(img_disp,
                        "fps = {:.1f}".format( (1.0 / (time.time() - self._prev_t) )),
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        self._prev_t = time.time()

    def humans_to_skels_list(self, humans, scale_h = None): 
        ''' Get skeleton data of (x, y * scale_h) from humans.
        Arguments:
            humans {a class returned by self.detect}
            scale_h {float}: scale each skeleton's y coordinate (height) value.
                Default: (image_height / image_widht).
        Returns:
            skeletons {list of list}: a list of skeleton.
                Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
            scale_h {float}: The resultant height(y coordinate) range.
                The x coordinate is between [0, 1].
                The y coordinate is between [0, scale_h]
            Changes:
            skeletons_x {list of lists}: a list of skeletons of x- axis.
            skeletons_y {list of lists}: a list of skeletons of y- axis.
        '''
        # if scale_h is None:
        #     scale_h = self._scale_h
        # skeletons = []
        # NaN = 0
        # for human in humans:
        #     skeleton = [NaN]*(18*2)
        #     for i, body_part in human.body_parts.items(): # iterate dict
        #         idx = body_part.part_idx
        #         skeleton[2*idx]=body_part.x
        #         skeleton[2*idx+1]=body_part.y * scale_h
        #     skeletons.append(skeleton)
        # return skeletons, scale_h
        if scale_h is None:
            scale_h = self._scale_h
        skeletons_x = []
        skeletons_y = []
        NaN = 0
        for human in humans:
            skeleton_x = [NaN]*(18*1)
            skeleton_y = [NaN]*(18*1)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton_x[idx]=body_part.x
                skeleton_y[idx]=body_part.y * scale_h
            skeletons_x.append(skeleton_x)
            skeletons_y.append(skeleton_y)
        return skeletons_x, skeletons_y, scale_h

     
def test_openpose_on_webcamera():
    '''Tess the fuctions on Web_Cam'''
    # -- Initialize web camera reader
    from utils.uti_images_io import Read_Images_From_Webcam, Image_Displayer
    Webcam_Reader = Read_Images_From_Webcam(fMax_Framerate=10)
    img_displayer = Image_Displayer()
    
    # -- Initialize openpose detector    
    skeleton_detector = Skeleton_Detector("mobilenet_thin", "432x368")

    # -- Read image and detect
    import itertools
    for i in itertools.count():
        img = Webcam_Reader.Read_Image()
        if img is None:
            break
        print(f"Read {i}th image...")

        # Detect
        humans = skeleton_detector.detect(img)
        
        # Draw
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp)
        lists = skeleton_detector.humans_to_skels_list(humans)
        print(lists)
    print("Program ends")

if __name__ == "__main__":
    test_openpose_on_webcamera()