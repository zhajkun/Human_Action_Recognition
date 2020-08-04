# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1: Add the function to write the skeleton data

"""
{
    A simple programm, extended from run.py from tf-openpose project. Just to test the syntax of using tf-openpose. The skeleton data 
    will be store in filepath = '/home/zhaj/tf_test/Test/Skeleton_Data'. Test the programm using:
    python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
    where 'p1.jpg' is the name of input image.
}

{}
"""
import argparse
import logging
import sys
import time
import utils.uti_commons as uti_common
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

skeletons = []
NaN = 0
for human in humans:
    skeleton = [NaN]*(18*2)
    for i, body_part in human.body_parts.items(): # iterate dict
        idx = body_part.part_idx
        skeleton[2*idx] = body_part.x
        skeleton[2*idx+1] = body_part.y 
        skeletons.append(skeleton)
