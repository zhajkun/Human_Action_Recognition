
import os, sys
import numpy as np

file_path = "C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_2.npz"

with np.load(file_path) as data:
    datasets_position = data['FEATURES_POSITION']
    datasets_velocity = data['FEATURES_VELOCITY']
    labels = data['FEATURES_LABELS']

indices = np.random.permutation(labels.shape[0])
valid_cnt = int(datasets_position.shape[0] * 0.3)
test_idx,training_idx=indices[:valid_cnt],indices[valid_cnt:]
test_pos, train_pos = datasets_position[test_idx,:], datasets_position[training_idx,:]
test_labels, train_labels = labels[test_idx], labels[training_idx]
test_vol, train_vol = datasets_velocity[test_idx,:], datasets_velocity[training_idx]

np.savez("C:/Users/Kun/tf_test/Human_Action_Recognition/data_proc/Data_Features/features_split.npz",
                    FEATURES_POSITION_TRAIN = train_pos, FEATURES_VELOCITY_TRAIN = train_vol, LABELS_TRAIN = train_labels,
                    FEATURES_POSITION_TEST = test_pos, FEATURES_VELOCITY_TEST = test_vol, LABELS_TEST = test_labels)

print('Finish')