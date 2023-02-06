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

## Project results
### Dataset
#### Dataset analysis
Dataset has various street images which includes vehicles, pedestrians and cyclists. Images were taken in the variou weather, time condition as you can see below. (vehicles in red, pedestrians in blue, cyclist in green)
inserire immagine del dataset
And I randomly selected 100 images and calculated the distribution of labels like below.
inserire immagine plot classi
### Training
#### Reference experiment
Result from the reference experiment is as follow.

immmagine loss

As you can see, the total loss is above 2.00 and I consider it's quite bad numbers.
AP numbers are also pretty bad like below and almost no object is detected.

'forse dire che la eval non trova nessun oggetto'

#### Improve on the reference

##### Experiment4

After several tries, I could make a breakthrough by adding some data augments options in [pipeline_new_augm.config](experiments/experiment4/pipeline_new.config) like below.

```
  data_augmentation_options {
      random_rgb_to_gray {
        probability: 0.3
      }
  }
  data_augmentation_options {
      random_distort_color {
        color_ordering: 1
      }
  }
  data_augmentation_options {
      random_adjust_brightness {
        max_delta: 0.4
      }
  }
  data_augmentation_options {
      random_adjust_contrast {
        min_delta: 0.8
        max_delta: 1.25
      }
  }
  data_augmentation_options {
      random_adjust_hue {
        max_delta: 0.04
      }
  }
  data_augmentation_options {
      random_adjust_saturation {
        min_delta: 0.8
        max_delta: 1.25
      }
  }
```

Those data augment could added more samples for cloudy or night scenes and it clearly help reducing the loss.

And the result loss is like below. (The pink lines indicate numbers from a new model.)

![experiment4_loss](pics/experiment4_loss.png)

But still AP is not enough and nothing was detected from the test datasets.
![experiment4_ap](pics/experiment4_ap.png)
