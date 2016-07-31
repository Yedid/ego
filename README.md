This repo contains code for our paper "An Egocentric Look at Video Photographer Identity" Y. Hoshen and S. Peleg, CVPR'16, Las Vegas, June 2016.

The paper can be found at: http://arxiv.org/abs/1411.7591

This code was tested on Ubuntu 14.04

Installation  

<<<<<<< HEAD
Our code has dependencies on keras and numpy. We suggest installing keras in a virtual env
$ cd ~
$ virtualenv keras
$ source ~/keras/bin/activate
$ pip install keras

Running the code

Assuming the repo is cloned in ~/ego
$ cd ~/ego/code
$ python preprocess.py
$ python train.py
=======
Our code has dependencies on keras and numpy. We suggest installing keras in a virtual env.  
$ cd ~  
$ virtualenv keras  
$ source ~/keras/bin/activate  
$ pip install keras  

Running the code

Assuming the repo is cloned in ~/ego   
$ cd ~/ego/code  
$ python preprocess.py  
$ python train.py  
>>>>>>> bec5878a203bc55afe75339456748a93502791d9

Note that the basic keras installation uses theano. This requires changing setting is_theano=True in code/train.py

The results are a few percentage points better than presented in the paper due to implementation improvements.

Dataset

Due to privacy issues we are not yet releasing the video dataset used in this paper. Anonymizing the dataset is ongoing work. Instead we present the optical flow data for each frame in our dataset. The data is presented in the /datasets directory.

Three datasets are presented. black - data collected by the GoPro 3+ Black camera, grey - data collected from our GoPro 3+ Silver camera, and last data recorded by the GoPro 3+ Silver camera at least a week after the initial data collected.

Inside each dataset appear directories for each one of the participants, the directory name is the participant ID. Each participant directory, contains one or more CSV files corresdponding to walking sequences recorded by the participants.

Each CSV file is structured with a row for each feature (first the optical flow X component for each block, and then the y component for each block). Each frame in the sequence is presente by a column in the CSV file.

To create a dataset of your own please conform to the same format. Blockwise optical flow can be computed by the excellent code presented here http://www.vision.huji.ac.il/egoseg/. Note that their code only runs on MS Windows. 

License

This code is released for research purposes only. For other uses please contact the authors.

Errata

Please report any bugs to ydidh@cs.huji.ac.il
