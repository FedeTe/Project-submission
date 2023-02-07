# Object Detection Project
In this project it was created a CNN to detect and classify objects using data from Waymo dataset and a SSD Resnet 50 640x640 model. At first some exploratory data analysis has been performed on the image batch  (Exploratory Data Analysis notebook), then the model has been trained and evaluated.

## Structure
### Experiments
The experiments folder is organized as follow:
```
Experiments/
    - Reference/ -  reference training
    - Exp_Augmentation/ - experiment on data augmentations
    - Exp_Optimization1/ - experiment on decay of learning rate
    - Exp_Optimization2/ - experiment on constant learning rate
    ...
```
## Prerequisites
For the project it was used the on-line Project Workspace - Jupyter Notebooks.

## Project Overview
### Dataset
Dataset has various street images which includes vehicles, pedestrians and cyclists. Images were taken in various weather, time condition as you can see below. (vehicles in red, pedestrians in blue, cyclist in green).

![dataset](Pic/data_vis.png)

Here I randomly selected 50 images and calculated the distribution of labels:

![class_distr](Pic/class_distr.png)

Taking this batch as a reference, most of the object in the dataset are vehicles and very few are cyclist.

## Training
### Reference experiment
The following image show the result of the training process (the reference pipeline file is here [pipeline_new.config](Experiments/Reference/pipeline_new.config)).

![before_imp](Pic/loss_before_improvement.png)

As you can see, the total loss is well above 2.00 and I consider it's quite bad numbers. 
In the following experiments I'll try to improve the performance.

## Improve on the reference
### Experiment on augmentation
After several tries, I was able to slightly improve performances with a combination on the following augment options in [pipeline_new_augm.config](Experiments/Exp_Augmentation/pipeline_new_augm.config)

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

And the result loss is like below.

![loss_after_augment](Pic/loss_after_augment.png)

Here some examples of augmented images:

![augmented_images](Pic/augmented_image.png)

### Experiment on optimization #1
Here's the pipeline config for this experiment: [pipeline_new_opt.config](Experiments/Exp_Oprimization/pipeline_new_opt.config)

Here the focus was on reducing the loss at the later part of training and added decaying of learning rate like below.

```
  optimizer {
    adam_optimizer {
      learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.001
          decay_steps: 700
        }
      }
    }
    use_moving_average: false
  }
```

![exp_opt_loss](Pic/loss_after_opt.png)

There's a good improvement on total loss after augmentation change, and it's more stable in the initial part of the epochs (even if seems noisier compared to previous loss but the scale is different)

### Experiment on optimization #2
Here's the pipeline config for this experiment: [pipeline_new_opt2.config](Experiments/Exp_Optimization2/pipeline_new_opt2.config)

I've tried a different optimization to see if the result can improve further.

```
  optimizer {
    adam_optimizer {
      learning_rate {
        constant_learning_rate {
          learning_rate: 0.001
        }
      }
    }
    use_moving_average: false
  }
```

Total loss slightly improved compared to ```Experiment on optimization #1```, it's lower and more stable even if there's a spyke around 1.2k step.

![exp_opt2_loss](Pic/loss_after_opt2.png)

