#! /bin/bash
#virtualenv venv --python=python3.6
#source venv/bin/activate
pip install -U torch==1.7.1 numpy==1.19.5
pip install -r requirements.txt

#Generate processed data for different segmentation object on S3DIS dataset.
#There are two arguments are required to be specified:
#1. Choose the 'obj' from 0~12 to reproduce the experimental results for different object.
#2. Choose the 'data' from ['train', 'test'] to generate the test set or the training set.
python3 ./dataset/gen_S3DIS.py --obj 2 --data 'train'

#Reproduce the experimental results of training the model with 80% of the nodes in synthetic data.
#There are two arguments:
#1. Choose the 'model_name' from ['GMGCN','ATTGCN'].
#2. Choose the 'hier' from ['W','WO'] to determine whether the model has a hierarchical graph structure.
python3 ./synthetic_model/main.py --model_name 'GMGCN' --hier 'W'

#Reproduce the experimental results on PHEME data.
#There are two arguments:
#1. Choose one event from ['charliehebdo','ferguson','germanwings-crash','ottawashooting','sydneysiege'].
#2. Choose the 'occur_times' that determines the user labels from [2,3,4].
python3 ./PHEME_model/main.py --data_name 'germanwings-crash' --occur_times 4

#Reproduce the experimental results of GMGCN on S3DIS dataset.
#There are three arguments:
#1. Choose the 'obj' from 0~12 to reproduce the experimental results on different object dataset.
#2. Choose the 'multi_gpus' that determines whether to use multiple GPUs for parallel computing.
#3. Choose the 'testdata' from ['train', 'test'] to determine whether to test the model on the test set or the training set.
python3 ./S3DIS_model/main.py --obj 2  --multi_gpus  --testdata 'train'

#deactivate