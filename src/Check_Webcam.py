# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 

"""
{
    Only to display the Webcam frames for prepare
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
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# […]

# Own modules
# from {path} import {class}
# […]

__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'