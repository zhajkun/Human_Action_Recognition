# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 1

"""
{
    Get skeletons data from source images, and rebuild the data by change the order of joints.
    This is the 1. Version of this module, which currently only support reading a skeleton of 1 person.
    The source images could be images inside a given folder, a vidoo in a folder or Web_Camera live.
    Read those images and processing it to output the rebuilded
    skeletons coordinates in x- and y- axis seperatly and the diaplacements between them as well
    Input:

    Output:

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
if True:  # Include project path

    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    
    # Own modules
    import utils.uti_images_io as uti_images_io
    import utils.uti_openpose as uti_openpose
    import utils.uti_skeletons_io as uti_skeletons_io
    import utils.uti_commons as uti_commons
# […]

'''TODO: Using yaml config file to define the input and output folder path!''' 
if True:
    # Temporal path and format, use config.yaml file to define it later.
    # SKELETON_X_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X/')  # Original skeletons data from tf-openpose, no need to save
    # SKELETON_Y_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_Y/')
    ''' The rubuilded skeletons data in x- and y- axis, which contains 35 joints, the new order of joints check descriptions file'''
    
    SKELETON_X_FOLDER_DIR = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_DIR/')
    SKELETON_Y_FOLDER_DIR = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_Y_DIR/')
    
    ''' The rubuilded skeletons displacements in x- and y- axis, which contains 35 joints, the new order of joints check descriptions file'''
    
    SKELETON_X_FOLDER_VEL = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_X_VEL/')
    SKELETON_Y_FOLDER_VEL = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Skeletons/Test_Skeleton_Y_VEL/')
    
    ''' Save the images coresponding to the skeletons for debug or visualization'''
    
    DST_VIZ_IMGS_FOLDER = ('/home/zhaj/tf_test/Human_Action_Recognition/Data_Images/Test_Images/')

    '''Define the saved files name format, start with 00000, have 5 digitals'''

    SKELETON_FILENAME_FORMAT = ('{:05d}.txt')
    IMAGE_FILENAME_FORMAT = ('{:05d}.jpg')
def Read_Skeletons_From_Web_Camera():
    '''Read and diaplay the images from web camera, save the skeletons '''
    import itertools
    for i in itertools.count():
        img = Data_Source.Read_Image()
        if img is None:
            break
        print(f"Read {i}th Frame from Web Camera...")

        Detected_Human = Skeleton_Detector.detect(img)
        Image_Output = img.copy()
        Skeleton_Detector.draw(Image_Output, Detected_Human)
        Image_Window.display(Image_Output)
        Skeleton_X, Skeleton_Y, Scale_h = Skeleton_Detector.humans_to_skels_list(Detected_Human)

        if Skeleton_X and Skeleton_Y: #only save non-empty lists 
            txt_filename = SKELETON_FILENAME_FORMAT.format(i)
            uti_commons.save_listlist(SKELETON_X_FOLDER + txt_filename, Skeleton_X)
            uti_commons.save_listlist(SKELETON_Y_FOLDER + txt_filename, Skeleton_Y)
            Skeleton_X_DIR = uti_skeletons_io.Rebuild_Skeletons(Skeleton_X)
            Skeleton_Y_DIR = uti_skeletons_io.Rebuild_Skeletons(Skeleton_Y)
            uti_commons.save_listlist(SKELETON_X_FOLDER_DIR + txt_filename, Skeleton_X_DIR)
            uti_commons.save_listlist(SKELETON_Y_FOLDER_DIR + txt_filename, Skeleton_Y_DIR)

            print(f"Saved {i}th Skeleton Data from Webcam...")
            jpg_filename = IMG_FILENAME_FORMAT.format(i)
            cv2.imwrite(DST_IMGS_FOLDER + jpg_filename, img)
            cv2.imwrite(DST_VIZ_IMGS_FOLDER + jpg_filename, Image_Output)
            print(f"Saved {i}th Image with Skeleton Data from Webcam...")


    print("Program ends")


# Main function, defaul to read images from web camera
if __name__ == "__main__":
    

__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
