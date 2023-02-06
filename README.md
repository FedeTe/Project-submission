# Object Detection Project
In this project it was created a CNN to detect and classify objects using data from Waymo dataset and a SSD Resnet 50 640x640 model. At first some exploratory data analysis has been performed on the image batch  (Exploratory Data Analysis notebook), then the model has been trained and evaluated.

The projects contains all the requested files plus a "Object_Deteciotn.pdf" with the analysis on project results. N.B. there're two different .config files, the provided one and the modified one used for the various augmentation trials (it contains the last trial)
## Structure
### Data
The data is organized as follow:
```
/home/workspace/data/
    - train: contain the train data
    - val: contain the val data
    - test - contains files to test the model and create inference videos
```
### Experiments
The experiments folder is organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file, the modified config file and all checkpoints and tfevent from the training and evaluation phase
    - label_map.pbtxt
    ...
```
## Prerequisites
For the project is was used the on-line Project Workspace - Jupyter Notebooks

# Project results
### Dataset
#### Dataset analysis
