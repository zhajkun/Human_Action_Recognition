# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version

"""
{
    Description
}
{License_info}
"""

# Futures

# […]

# Built-in/Generic Imports
import os
import sys
import numpy as np
# […]

# Libs
# import pandas as pd # Or any other
# […]

# Own modules
# from {path} import {class}
# […]
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

def delete_invalid_skeletons_from_lists(skeletons_src):
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
    if not skeletons_src:
        return []

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
        if iValid_Joints >= 8 and fLength_of_Y >= 0.25 and iValid_Key_Joints >= 4:
            skeletons_dir.append(skeleton)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   
            # add this skeleton only when all requirements are satisfied
            
    return skeletons_dir

class Tracker(object):
    ''' A simple tracker:

        For previous skeletons(S1) and current skeletons(S2),
        S1[i] and S2[j] are matched, if:
        1. For S1[i],   S2[j] is the most nearest skeleton in S2.
        2. For S2[j],   S1[i] is the most nearest skeleton in S1.
        3. The distance between S1[i] and S2[j] are smaller than self._dist_thresh.
            (Unit: The image width is 1.0, the image height is scale_h=rows/cols)

        For unmatched skeletons in S2, they are considered 
            as new people appeared in the video.
    '''

    def __init__(self, dist_thresh=0.4, max_humans=5):
        ''' 
        Arguments:
            dist_thresh {float}: 0.0~1.0. The distance between the joints
                of the two matched people should be smaller than this.
                The image width and height has a unit length of 1.0.
            max_humans {int}: max humans to track.
                If the number of humans exceeds this threshold, the new
                skeletons will be abandoned instead of taken as new people.
        '''
        self._dist_thresh = dist_thresh
        self._max_humans = max_humans
        self._ref_point = [0.5, 0.5]
        self._dict_id2skeleton = {}
        self._cnt_humans = 0

    def track(self, curr_skels):
        ''' Track the input skeletons by matching them with previous skeletons,
            and then obtain their corresponding human id. 
        Arguments:
            curr_skels {list of list}: each sub list is a person's skeleton.
        Returns:
            self._dict_id2skeleton {dict}:  a dict mapping human id to his/her skeleton.
        '''

        curr_skels = self._sort_skeletons_by_dist_to_center(curr_skels)
        N = len(curr_skels)

        # Match skeletons between curr and prev
        if len(self._dict_id2skeleton) > 0:
            ids, prev_skels = map(list, zip(*self._dict_id2skeleton.items()))
            good_matches = self._match_features(prev_skels, curr_skels)

            self._dict_id2skeleton = {}
            is_matched = [False]*N
            for i2, i1 in good_matches.items():
                human_id = ids[i1]
                self._dict_id2skeleton[human_id] = np.array(curr_skels[i2])
                is_matched[i2] = True
            unmatched_idx = [i for i, matched in enumerate(
                is_matched) if not matched]
        else:
            good_matches = []
            unmatched_idx = range(N)

        # Add unmatched skeletons (which are new skeletons) to the list
        num_humans_to_add = min(len(unmatched_idx),
                                self._max_humans - len(good_matches))
        for i in range(num_humans_to_add):
            
            self._dict_id2skeleton[self._cnt_humans] = np.array(
                curr_skels[unmatched_idx[i]])
            self._cnt_humans += 1
        return self._dict_id2skeleton

    def _get_neck(self, skeleton):
        x, y = skeleton[2], skeleton[3]
        return x, y

    def _sort_skeletons_by_dist_to_center(self, skeletons_src):
        ''' Skeletons are sorted based on the distance
        between neck and image center, from small to large.
        A skeleton near center will be processed first and be given a smaller human id.
        Here the center is defined as (0.5, 0.5), although it's not accurate due to h_scale.
        '''
        skeletons_src.sort(key = lambda P: (P[2] - self._ref_point[0])**2 + (P[3] - self._ref_point[1])**2)
        skeletons_sorted = skeletons_src
        return skeletons_sorted

    def _match_features(self, features1, features2):
        ''' Match the features.　Output the matched indices.
        Returns:
            good_matches {dict}: a dict which matches the 
                `index of features2` to `index of features1`.
        '''
        features1, features2 = np.array(features1), np.array(features2)

        #cost = lambda x1, x2: np.linalg.norm(x1-x2)
        def calc_dist(p1, p2): return (
            (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        def cost(sk1, sk2):

            # neck, shoulder, elbow, hip, knee
            joints = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12,
                               13, 16, 17, 18, 19, 22, 23, 24, 25])

            sk1, sk2 = sk1[joints], sk2[joints]
            valid_idx = np.logical_and(sk1 != 0, sk2 != 0)
            sk1, sk2 = sk1[valid_idx], sk2[valid_idx]
            sum_dist, num_points = 0, int(len(sk1)/2)
            if num_points == 0:
                return 99999
            else:
                for i in range(num_points):  # compute distance between each pair of joint
                    idx = i * 2
                    sum_dist += calc_dist(sk1[idx:idx+2], sk2[idx:idx+2])
                mean_dist = sum_dist / num_points
                mean_dist /= (1.0 + 0.05*num_points)  # more points, the better
                return mean_dist

        # If f1i is matched to f2j and vice versa, the match is good.
        good_matches = {}
        n1, n2 = len(features1), len(features2)
        if n1 and n2:

            # dist_matrix[i][j] is the distance between features[i] and features[j]
            dist_matrix = [[cost(f1, f2) for f2 in features2]
                           for f1 in features1]
            dist_matrix = np.array(dist_matrix)

            # Find the match of features1[i]
            matches_f1_to_f2 = [dist_matrix[row, :].argmin()
                                for row in range(n1)]

            # Find the match of features2[i]
            matches_f2_to_f1 = [dist_matrix[:, col].argmin()
                                for col in range(n2)]

            for i1, i2 in enumerate(matches_f1_to_f2):
                if matches_f2_to_f1[i2] == i1 and dist_matrix[i1, i2] < self._dist_thresh:
                    good_matches[i2] = i1

            if 0:
                print("distance matrix:", dist_matrix)
                print("matches_f1_to_f2:", matches_f1_to_f2)
                print("matches_f1_to_f2:", matches_f2_to_f1)
                print("good_matches:", good_matches)

        return good_matches

if __name__=='__main__':

    localtracker = Tracker()
    
    testskls = [[0.8, 0.8, 0.649390243902439, 0.21195652173913043, 0.6097560975609756, 0.2078804347826087, 
    0.5914634146341463, 0.28125, 0.5853658536585366, 0.3342391304347826, 0.6890243902439024, 
    0.2078804347826087, 0.7042682926829268, 0.27309782608695654, 0.7042682926829268, 0.3342391304347826, 
    0.6128048780487805, 0.34646739130434784, 0.6128048780487805, 0.4402173913043479, 0.600609756097561, 
    0.5298913043478262, 0.6707317073170732, 0.34646739130434784, 0.6707317073170732, 0.4442934782608695,
     0.6798780487804879, 0.5380434782608695, 0.6402439024390244, 0.13858695652173914, 0.6554878048780488, 0.13858695652173914, 
     0.6310975609756098, 0.15081521739130435, 0.6707317073170732, 0.15081521739130435], [0.7103658536585366, 0.5141304347826087, 0.6341463414634146,
      0.1671195652173913, 0.6097560975609756, 0.15896739130434784, 0.625, 0.26902173913043476, 0.7378048780487805, 
      0.24864130434782605, 0.6554878048780488, 0.17119565217391303, 0.676829268292683, 0.25271739130434784, 0, 0, 0.6189024390243902, 0.37092391304347827, 
      0.6310975609756098, 0.5584239130434783, 0, 0, 0.6646341463414634, 0.36277173913043476, 0, 0, 0, 0, 0.7042682926829268, 0.09375, 0.7195121951219512, 
      0.10190217391304347, 0.6585365853658537, 0.08152173913043478, 0, 0], [0.7962962962962963, 0.12228260869565218, 0.7314814814814815, 0.1875, 0.6759259259259259, 
      0.17934782608695654, 0.6898148148148148, 0.29347826086956524, 0.8148148148148148, 0.4157608695652174,
       0.7870370370370371, 0.19565217391304346, 0.7916666666666666, 0.28125, 0.8240740740740741, 0.3831521739130435, 
       0.6111111111111112, 0.36684782608695654, 0.6388888888888888, 0.5828804347826086, 0, 0, 0.6805555555555556, 
       0.37907608695652173, 0.6435185185185185, 0.5828804347826086, 0, 0, 0.7824074074074074, 0.09782608695652173, 
       0.8194444444444444, 0.11005434782608697, 0.7407407407407407, 0.09375, 0, 0], [0.47685185185185186, 0.14266304347826086, 0.37037037037037035,
        0.14673913043478262, 0.35648148148148145, 0.14266304347826086, 0.4444444444444444, 0.27717391304347827, 
        0.5555555555555556, 0.26902173913043476, 0.3888888888888889, 0.15081521739130435, 0, 0, 0, 0, 
        0.25925925925925924, 0.3994565217391305, 0, 0, 0, 0, 0.35648148148148145, 0.37092391304347827, 
        0, 0, 0, 0, 0.47685185185185186, 0.12228260869565218, 0, 0, 0.4351851851851852, 0.10597826086956522, 0, 0] ]
    
    test_skl_t2 = [[0.8425925925925926, 0.17527173913043478, 0.7268518518518519, 0.17527173913043478, 0.7268518518518519, 
    0.1875, 0.7129629629629629, 0.28532608695652173, 0.7777777777777778, 0.42391304347826086, 0.7268518518518519, 0.16304347826086957, 
    0, 0, 0, 0, 0.6111111111111112, 0.36277173913043476, 0.6111111111111112, 0.5788043478260869, 0.6157407407407407, 0.7296195652173914, 
    98148148148148, 0.37907608695652173, 0.6620370370370371, 0.5706521739130435, 0.6296296296296297, 0.7255434782608695, 
    0.8425925925925926, 0.15489130434782608, 0.8657407407407407, 0.1671195652173913, 0.8009259259259259, 0.12228260869565218, 0, 0]]
    
    test_skl_t3 = [[0.8425925925925926, 0.17527173913043478, 0.7268518518518519, 0.17527173913043478, 0.7268518518518519, 
    0.1875, 0.7129629629629629, 0.28532608695652173, 0.7777777777777778, 0.42391304347826086, 0.7268518518518519, 0.16304347826086957, 
    0, 0, 0, 0, 0.6111111111111112, 0.36277173913043476, 0.6111111111111112, 0.5788043478260869, 0.6157407407407407, 0.7296195652173914, 
    98148148148148, 0.37907608695652173, 0.6620370370370371, 0.5706521739130435, 0.6296296296296297, 0.7255434782608695, 
    0.8425925925925926, 0.15489130434782608, 0.8657407407407407, 0.1671195652173913, 0.8009259259259259, 0.12228260869565218, 0, 0]]
    
    test_skl_t4 = [[0.8425925925925926, 0.17527173913043478, 0.7314814814814815, 0.17527173913043478, 0.7222222222222222, 0.18342391304347827, 0.7037037037037037, 0.27717391304347827, 0.7731481481481481, 0.4279891304347826, 0.7407407407407407, 0.17527173913043478, 0.7314814814814815, 0.27717391304347827, 0, 0, 0.6111111111111112, 0.3586956521739131, 0.6157407407407407, 0.5828804347826086, 0.6157407407407407, 0.7296195652173914, 0.6990740740740741, 0.3586956521739131, 0.6574074074074074, 0.5747282608695652, 0.6157407407407407, 0.7296195652173914, 0.8425925925925926, 0.15081521739130435, 0.8611111111111112, 0.1671195652173913, 0.7962962962962963, 0.12228260869565218, 0, 0]]
    
    test_skl_t5 = [[0.8425925925925926, 0.17527173913043478, 0.7268518518518519, 0.17527173913043478, 0.7268518518518519, 
    0.1875, 0.7129629629629629, 0.28532608695652173, 0.7777777777777778, 0.42391304347826086, 0.7268518518518519, 0.16304347826086957, 
    0, 0, 0, 0, 0.6111111111111112, 0.36277173913043476, 0.6111111111111112, 0.5788043478260869, 0.6157407407407407, 0.7296195652173914, 
    98148148148148, 0.37907608695652173, 0.6620370370370371, 0.5706521739130435, 0.6296296296296297, 0.7255434782608695, 
    0.8425925925925926, 0.15489130434782608, 0.8657407407407407, 0.1671195652173913, 0.8009259259259259, 0.12228260869565218, 0, 0]]
    
    test_skl_t6 = [[0.1]*36]

    test_skl_t7 = [[0.1]*36]
    test_skl_t8 = [[0.1]*36]
    test_skl_t9 = [[0.1]*36]
    test_skl_t10 = [[0.1]*36]
    test_skl_t11 = [[0.1]*36]
    
    dicttemp = {}

    dicttemp = localtracker.track(testskls)

    dicttemp = localtracker.track(test_skl_t2)

    dicttemp = localtracker.track(test_skl_t3)

    dicttemp = localtracker.track(test_skl_t4)

    dicttemp = localtracker.track(test_skl_t5)

    dicttemp = localtracker.track(test_skl_t6)

    dicttemp = localtracker.track(test_skl_t7)

    dicttemp = localtracker.track(test_skl_t8)

    dicttemp = localtracker.track(test_skl_t9)
    #skeletons_dir = delete_invalid_skeletons_from_lists(testskls)
    print(dicttemp)
    
