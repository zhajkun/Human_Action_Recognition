# Human_Action_Recognition
This is the Master-Theis of Jiakun Zhang. If you have any questions, please contact me.<br>
E-Mail: zhajkun@gmail.com
![GitHub](https://github.com/zhajkun/Human_Action_Recognition/blob/master/document/ntu_example.gif)<br>
# Prerequisites
The following code is based on Python 3.6.8, and TensorFlow 1.15. 
# Installation
1. Fisrt, you need to intall tf-pose-estimation. This is a OpenPose implemented using Tensorflow.<br>
https://github.com/ildoonet/tf-pose-estimation <br>
Please remember the location of your local tf-pose-estimation, you need to change this path at [config.json](./config/config.json)
2. Clone this repository to your local PC.
3. Run [requirements.txt](./requirements.txt) to install all dependencies.
4. Download and unzip datasets:<br>
The own recorded dataset:<br>
NTU RGB+D dataset:<br>
If you wish to access the whole NTU RGB+D dataset, please contact me or apply for it from:<br>
http://rose1.ntu.edu.sg/datasets/requesterAdd.asp?DS=3
5. Download other trained model (optional).<br>
http://rose1.ntu.edu.sg/datasets/requesterAdd.asp?DS=3  <br>
The model already in repository taking 20 frames as input, and the configuration files [config.json](./config/config.json) are setted to run this model. <br>
If you want to try other trainde models, rember to change configurations. <br>
6. At this point. the system should be ready to run. You can use [test_network_on_webcam.py](./src/test_network_on_webcam.py) to test it.
# Program structure
The whole Structure of this repository is showed bellow:<br>
![GitHub](https://github.com/zhajkun/Human_Action_Recognition/blob/master/document/programm_structur_2.png)<br>
At first, like mentioned before. Make sure the [config.json](./config/config.json) is adjusted for your PC. Especially check if the file path is right. <br>
The red blocks are libraries, blue are main scripts and green are data. 
# How to use
At the begining of all programs you will find detail instructions. This introduction will give you a overview of all programs. 
# Record own dataset
If you wish to record your own dataset, start with [Images_Recorder.py](./tools/Images_Recorder.py). Default FPS is 10, change it in [uti_images_io.py](./utils/uti_images_io.py) if you need.<br>
The out puts are images. Format is .jpg. All images are numbered from 00000.jpg. And the folder is named after the action class and time stample of recording. <br>
This formation will help to process images later.
# Get skeletons from exisitng dataset
If you wish to get skeletons from exisitng dataset (videos, images), use [s1_get_skeletons_data.py](./src/s1_get_skeletons_data.py) first.<br>
Remember change the source file path from [config.json](./config/config.json). <br>
You will recive all skeletons from images. The possible noise are already filter at this point.<br>
You can change the thresholds of filter from [uti_tracker.py](./utils/uti_tracker.py)<br>
The outputs of this program are txt files. <br>
Those txt files will be conert to numpy array and pack together by [s2_pack_all_text_files.py](./src/s2_pack_all_text_files.py).<br>
Of course you can write your own program to convert it to .csv files.<br>
The input features of the CNN network are caculated by [s3_pre_processing.py](./src/s3_pre_processing.py).<br>
You can also modife here to generate your own features with different time length. <br>
# Train the network
If you have the datasets and fetures ready, run [train_network.py](./src/train_network.py) to start tranining. <br>
The hyperparameter could be changed via [config.json](./config/config.json).<br>
**!!!!!IMPORTANT!!!!!**<br>
[config.json](./config/config.json) have an important parameter, the [FEATURE_WINDOW_SIZE].<br>
This is how many frames are in one clip, also the time length of one clip. <br>
Please remember the feature size and the input of network are matched.<br>
For example, if you are using the network of 20 frames, the [FEATURE_WINDOW_SIZE] mus  also be 20.<br>
# Test the network
If you want to run a live demo on webcam, videos, or images, just run [test_network_on_webcam.py](./src/test_network_on_webcam.py).<br>
The specific instructions are inside the program.<br>
If you want to run test on NTU RGB+D dataset, use [test_network_on_videos.py](./src/test_network_on_videos.py).<br>
