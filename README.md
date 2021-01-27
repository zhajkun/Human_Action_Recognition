# Human_Action_Recognition
This is the Master-Theis of Jiakun Zhang. If you have any questions, please contact me.<br>
E-Mail: zhajkun@gmail.com
# Prerequisites
The following code is based on Python 3.6.8, and TensorFlow 1.15. 
# Installation
1. Fisrt, you need to intall tf-pose-estimation. This is a OpenPose implemented using Tensorflow.<br>
https://github.com/ildoonet/tf-pose-estimation <br>
Please remember the location of your local tf-pose-estimation, you need to change this path at [config.json](./config/config.json)
2. Clone this repository to your local PC.
3. Run requirements.txt to install all dependencies.
4. Download and unzip datasets:<br>
The own recorded dataset:<br>
NTU RGB+D dataset:<br>
If you wish to access the whole NTU RGB+D dataset, please contact me or apply for it from:<br>
http://rose1.ntu.edu.sg/datasets/requesterAdd.asp?DS=3
5. Download trained model.<br>
http://rose1.ntu.edu.sg/datasets/requesterAdd.asp?DS=3  <br>
This model is taking 20 frames as input, and the configuration files [config.json](./config/config.json) are setted to run this model. <br>
If you want to try other trainde models, please contact me. <br>
6. At this point. the system should be ready to run. You can use [test_network_on_webcam.py](./src/test_network_on_webcam.py) to test it.
# Program Structure
