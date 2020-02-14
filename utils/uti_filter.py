# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Define the filter to different the skeletons in one image and track them in different images and discard the invalid skeletons
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import math
import functools
import numpy as np
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
# […]    mnist = tf.keras.datasets.mnist

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

# class Skeleton_Tracker(object):
    '''Skeleton Tracker: first label different skeletons in one image 
        and then try to track them and discard the invalid skeletons in the next images

    '''
#    def __init__(self):
def delete_invalid_skeletons_from_dict(skeletons_src):
    '''A simple function to delete invalid skeletons from the list of lists,
    thos skeletons without the key joints will be consider as invalid.
    Those key joints are: 
    N
    Arguments:
        skeletons_src {list of list}: the input list of lists, which is the proginal out put from tf-openpose
    Returns:
        skeletons_dir {list of list}: list without invalid lists
    '''
    skeletons_dir = []
    for skeleton in skeletons_src:
        s_x = skeleton[::2]
        s_y = skeleton[1::2]
        s_y = list(filter((0).__ne__, s_y))
        iValid_Joints = len([x for x in s_x if x != 0])
        # Neck – 1, Right Shoulder – 2, Right Elbow – 3, Left Shoulder – 5, Left Elbow – 6,
        # Right Hip – 8, Left Hip – 11
        key_joints = np.array([1, 2, 3, 5, 6, 8, 11])
        s_x = np.array(s_x)
        s_x = s_x[key_joints]
        iValid_Key_Joints = len([x for x in s_x if x != 0])
        fLength_of_Y = max(s_y) - min(s_y)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if iValid_Joints >= 8 and fLength_of_Y >= 0.25 and iValid_Key_Joints >= 5:
            # add this skeleton only when all requirements are satisfied
            skeletons_dir.append(skeleton)
    return skeletons_dir



class Skeleton_Tracker(object):
    ''' A simple tracker:

        For previous skeletons(S1) and current skeletons(S2),
        S1[i] and S2[j] are matched, if:
        1. For S1[i],   S2[j] is the most nearest skeleton in S2.
        2. For S2[j],   S1[i] is the most nearest skeleton in S1.
        3. The distance_to_origin between S1[i] and S2[j] are smaller than self._fDistance_max.
            (Unit: The image width is 1.0, the image height is scale_h=rows/cols)

        For unmatched skeletons in S2, they are considered 
            as new people appeared in the video.
    '''

    def __init__(self, fDistance_max=0.4, iHumans_max=5):
        ''' 
        Arguments:
            fDistance_max {float}: 0.0~1.0. The distance_to_origin between the joints
                of the two matched people should be smaller than this.
                The image width and height has a unit length of 1.0.
            iHumans_max {int}: max humans to track.
                If the number of humans exceeds this threshold, the new
                skeletons will be abandoned instead of taken as new people.
        '''
        self._fDistance_max = fDistance_max
        self._iHumans_max = iHumans_max

        self._lSkeletons_Output = {}
        self._iHumans_Counter = 0

    def track(self, skeletons_curr):
        ''' Track the input skeletons by matching them with previous skeletons,
            and then obtain their corresponding human id. 
        Arguments:
            skeletons_curr {list of list}: each sub list is a person's skeleton.
        Returns:
            self._lSkeletons_Output {dict}:  a dict mapping human id to his/her skeleton.
        '''

        skeletons_curr = self._sort_skeletons_by_distance_to_origin(skeletons_curr)

        iNum = len(skeletons_curr) # number of skeletons in current list

        # Match skeletons between current and previous
        if len(self._lSkeletons_Output) > 0:
            iIDs, skeletons_prev = map(list, zip(*self._lSkeletons_Output.items()))
            matched_skeletons = self._match_key_joints(skeletons_prev, skeletons_curr)

            self._lSkeletons_Output = {}
            bMatched_List = [False]*iNum
            for i2, i1 in matched_skeletons.items():
                human_id = iIDs[i1]
                self._lSkeletons_Output[human_id] = np.array(skeletons_curr[i2])
                bMatched_List[i2] = True
            unmatched_idx = [i for i, matched in enumerate(
                bMatched_List) if not matched]
        else:
            matched_skeletons = []
            unmatched_idx = range(iNum)

        # Add unmatched skeletons (which are new skeletons) to the list
        num_humans_to_add = min(len(unmatched_idx),
                                self._iHumans_max - len(matched_skeletons))
        for i in range(num_humans_to_add):
            self._iHumans_Counter += 1
            self._lSkeletons_Output[self._iHumans_Counter] = np.array(
                skeletons_curr[unmatched_idx[i]])

        return self._lSkeletons_Output

    def _get_neck_position(self, skeleton):
        ''' Extract the coordinates of the neck from skeletons list
        Arguments:
            skeleton {list}: a person's skeleton.
        Returns:
            x,y {float}: the coordinates of the neck on x- and y- axis.
        '''
        x, y = skeleton[2], skeleton[3]
        return x, y

    def _sort_skeletons_by_distance_to_origin(self, skeletons):
        ''' Skeletons are sorted based on the distance_to_origin
        between neck and image origin, from small to large.
        A skeleton near origin will be processed first and be given a smaller human id.
        '''
        def find_the_distance_between_two_points(p1, p2): 
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        def distance_to_origin(skeleton):
            x1, y1 = self._get_neck_position(skeleton)
            return find_the_distance_between_two_points((x1, y1), (0.0, 0.0))  # return the distance_to_origin

        def cmp(a, b): 
            return (a > b)-(a < b)
        
        def mycmp(sk1, sk2): 
            return cmp(distance_to_origin(sk1), distance_to_origin(sk2))
        
        sorted_skeletons = sorted(skeletons, key = functools.cmp_to_key(mycmp))
        
        return sorted_skeletons

    def _match_key_joints(self, skeletons_prev, skeletons_curr):
        ''' Match the key joints.　Output the matched indices.
        Returns:
            matched_skeletons {dict}: a dict which matches the 
                `index of features2` to `index of features1`.
        '''
        skeletons_prev, skeletons_curr = np.array(skeletons_prev), np.array(skeletons_curr)

        def find_the_distance_between_two_points(p1, p2): 
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.05

        def find_distance_between_key_joints(sk1, sk2):

            # Neck – 1, Right Shoulder – 2, Right Elbow – 3, Left Shoulder – 5, Left Elbow – 6,
            # Right Hip – 8, Left Hip – 11 index = 2*n and 2*n + 1
            key_joints = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12,
                               13, 16, 17, 22, 23])

            sk1, sk2 = sk1[key_joints], sk2[key_joints] # the list will be convert 14 bit 0-13

          
            
            ######################################################################################
            sum_dist, num_points = 0, int(len(sk1)/2)
            if num_points == 0:
                return 99999        
            else:
                for i in range(num_points):  # compute distance_to_origin between each pair of joint
                    idx = i * 2
                    sum_dist += find_the_distance_between_two_points(sk1[idx:idx+2], sk2[idx:idx+2])
                mean_dist = sum_dist / num_points
                mean_dist /= (1.0 + 0.05*num_points)  # more points, the better
                return mean_dist
            ##########################################################################################
        # If f1i is matched to f2j and vice versa, the match is good.
        matched_skeletons = {}
        n1, n2 = len(skeletons_prev), len(skeletons_curr)
        if n1 and n2:

            # distance_matrix[i][j] is the distance_to_origin between features[i] and features[j]
            distance_matrix = [[find_distance_between_key_joints(f1, f2) for f2 in skeletons_curr]
                           for f1 in skeletons_prev]
            distance_matrix = np.array(distance_matrix)

            # Find the match of features1[i]  
            matches_f1_to_f2 = [distance_matrix[row, :].argmin()
                                for row in range(n1)]

            # Find the match of features2[i]
            matches_f2_to_f1 = [distance_matrix[:, col].argmin()
                                for col in range(n2)]

            for i1, i2 in enumerate(matches_f1_to_f2):
                if matches_f2_to_f1[i2] == i1 and distance_matrix[i1, i2] < self._fDistance_max:
                    matched_skeletons[i2] = i1

            if 0:
                print("distance_to_origin matrix:", distance_matrix)
                print("matches_f1_to_f2:", matches_f1_to_f2)
                print("matches_f1_to_f2:", matches_f2_to_f1)
                print("matched_skeletons:", matched_skeletons)

        return matched_skeletons

if __name__ == "__main__":
    # sksletons = [[0,0,1,1],[0,0,0.1,0.1],[0,0,0.5,0.5],[0,0,0.4,0.4],[0,0,0.2,0.2],[0,0,0.25,0.25],
    # [0,0,0.61,0.61],[0,0,0.71,0.71],[0,0,0.75,0.75]]
    skeletons = [[0.25609756097560976, 0.14266304347826086, 0.2530487804878049, 0.21603260869565216, 
    0.20121951219512196, 0.21195652173913043, 0.18292682926829268, 0.2975543478260869, 0.17073170731707318, 
    0.37092391304347827, 0.3048780487804878, 0.21603260869565216, 0.3231707317073171, 0.2975543478260869, 
    0.3231707317073171, 0.37907608695652173, 0.21341463414634146, 0.375, 0.21646341463414634, 0.5013586956521738, 
    0.21036585365853658, 0.6277173913043479, 0.2804878048780488, 0.375, 0.3018292682926829, 0.5013586956521738, 
    0.31402439024390244, 0.6277173913043479, 0.24390243902439024, 0.13043478260869565, 0.2652439024390244, 0.13043478260869565,
     0.22865853658536586, 0.14266304347826086, 0.28353658536585363, 0.14266304347826086], [0.29878048780487804, 
     0.3546195652173913, 0.29878048780487804, 0.37907608695652173, 0.2896341463414634, 0.375, 
     0, 0, 0, 0, 0.31097560975609756, 0.37907608695652173, 0.31097560975609756, 0.4157608695652174,
      0.3048780487804878, 0.4279891304347826, 0.28353658536585363, 0.4483695652173913, 0.3231707317073171, 
      0.46467391304347827, 0.29878048780487804, 0.5380434782608695, 0.29878048780487804, 0.4483695652173913, 
      0.32926829268292684, 0.46467391304347827, 0.3231707317073171, 0.5339673913043479, 0.29878048780487804, 
      0.35054347826086957, 0.3018292682926829, 0.35054347826086957, 0, 0, 0.3079268292682927, 0.3546195652173913]]


     
    #  [0.25609756097560976, 
    #  0.13858695652173914, 0.0609756097560976, 0.01195652173913043, 0.2073170731707317, 0.21195652173913043, 
    #  0.18597560975609756, 0.29347826086956524, 0.17378048780487804, 0.37092391304347827, 0.3079268292682927, 
    #  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    #  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    #  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #   0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    # Tracker = Skeleton_Tracker()
    # sortedlist = Tracker._sort_skeletons_by_distance_to_origin(skeletons)
    afterfilter = delete_invalid_skeletons_from_dict(skeletons)
    print(afterfilter)



__author__ = '{author}'
__copyright__ = 'Copyright {year}, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '{mayor}.{minor}.{rel}'
__maintainer__ = '{maintainer}'
__email__ = '{contact_email}'
__status__ = '{dev_status}'
