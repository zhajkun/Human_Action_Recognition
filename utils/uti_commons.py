# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 

"""
{Description}
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
# […]

# Own modules
# from {path} import {class}
# […]


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

def saveliststofile(filepath, ll):
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in ll:
            f.write("%s\n" % item)

def readlists_nosimp(filepath):
    ''' Read a list of lists from file '''
    with open(filepath, 'r') as f:
       # ll = simplejson.load(f)
       ll = f.read().splitlines()
    return ll


