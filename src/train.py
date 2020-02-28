# temporal version, for one stream only
import numpy as np
import json
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    from utils.lib_classifier import ClassifierOfflineTrain



def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings

with open(ROOT + 'config/config.json') as json_config_file:
    config_all = json.load(json_config_file)
    config = config_all["train.py"]

    # common settings

    CLASSES = np.array(config_all["classes"])
    IMAGE_FILE_NAME_FORMAT = config_all["IMAGE_FILE_NAME_FORMAT"]
    SKELETON_FILE_NAME_FORMAT = config_all["SKELETON_FILE_NAME_FORMAT"]
    IMAGES_INFO_INDEX = config_all["IMAGES_INFO_INDEX"]

    # openpose



    # input

    # output
    
    MODEL_PATH = par(config["output"]["MODEL_PATH"])

# -- Functions

def train_test_split(X, Y, ratio_of_test_size):
    ''' Split training data by ratio '''
    IS_SPLIT_BY_SKLEARN_FUNC = True

    # Use sklearn.train_test_split
    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    # Make train/test the same.
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y

def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Evaluate accuracy and time cost '''

    # Accuracy
    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.5f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = lib_plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()



# -- Main


def main():

    # -- Load preprocessed data
    print("\nReading npy files of classes, features, and labels ...")

    SRC_PROCESSED_FEATURES = 'data_proc/Data_Features/features.npz'

    data = np.load(SRC_PROCESSED_FEATURES)
    train_examples = data['FEATURES_POSITION']
    v_exmaples = data['FEATURES_VELOCITY']
    train_labels = data['FEATURES_LABELS']

   
    # X = np.concatenate(train_examples, dtype=v_exmaples)  # features
    # Y = train_labels  # labels
    
    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        train_examples, train_labels, ratio_of_test_size=0.3)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # # -- Train the model
    print("\nStart training model ...")
    model = ClassifierOfflineTrain()
    model.train(tr_X, tr_Y)

    # # -- Evaluate model
    print("\nStart evaluating model ...")
    evaluate_model(model, CLASSES, tr_X, tr_Y, te_X, te_Y)

    # # -- Save model
    print("\nSave model to " + MODEL_PATH)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
