import functools

# import numpy as np
# import tensorflow as tf


# train_file_path = "C:/Users/Kun/tf_test/Human_Action_Recognition/iRobeka_Actions/Bending_Bottom_Shelf_1/2D_Skeleton_Bending_Bottom_Shelf_1_1.csv"

# # 让 numpy 数据更易读。
# np.set_printoptions(precision=3, suppress=True)


# CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

# def get_dataset(file_path):
#     dataset = tf.data.experimental.make_csv_dataset(
#         file_path, batch_size=12,
        
#         na_value="?",
#         num_epochs=1,
#         ignore_errors=True)

#     return dataset

# train_dataset = get_dataset(train_file_path)
import itertools
import os
import matplotlib.pyplot as plt
import tensorflow as tf
tf.executing_eagerly()

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

# CSV文件中列的顺序
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))

print(features)

plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()