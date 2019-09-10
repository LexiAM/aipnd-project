# Image Classification With PyTorch Transfer Learning Project

## Summary
In this project, we create a Python application for an image classifier built using transfer learning with PyTorch. Code development can be followed in the "Image Classifier Project.ipynb" Jupyter notebook (also available in html).
The classifier model is based on one of pretrained on ImageNet torchvision models with final fully connected classifier layers specified by user.

This project is part of the Udacity Introduction to Machine Learning Nanodegree.

## Project Structure
Main application modules are:
  - train.py : training a classifier model
  - predict.py : classifying images using a trained classifier model checkpoint

Support modules:
  - model_utils.py : helper functions for building, loading, saving, training and visualizing training history of a classifier model
  - data_utils.py : helper functions for loading, processing and transforming data, and visualizing prediction results
  - workslace_utils.py : helper functions for maintaining active workspace during long training cycles
  - cat_to_name.json : dictionary containing mapping of image class indices to real class names

## Details
### train.py
#### Description:
Trains a model with user specified classifier and saves model and optimizer parameters achieved after training epoch with the highest validation accuracy.

#### Options:
- data_directory : location of training, validation and test data
- -h, --help : train.py options help
- --save_dir : directory for saving model checkpoints
- --resume_checkpoint : filepath for model checkpoint for resumption of training.
                      If resume_checkpoint is specified --arch, --learning_rate,
                      --hidden_units, and --drop_p arguments are ignored.
- --arch : baseline torchvision model architecture from the following choices: alexnet, densenet161, resnet18, vgg16
- --learning_rate : training learning rate
- --hidden_units : list of hidden layer sizes separated by spaces
- --drop_p : dropout probability
- --epochs : number of training epochs
- --gpu : Use gpu for training, if available

#### Usage:
- basic usage:
`python train.py 'flowers/'`

- initial training:
`python train.py 'flowers/' --save_dir 'checkpoints/' --arch vgg16 --hidden_units 2048 512 --learning_rate 0.001 --epochs 5 --gpu`

- resumption of training from previous checkpoint:
`python train.py 'flowers/' --resume_checkpoint 'checkpoints/checkpoint1.pth' --epochs 5 --gpu`

### predict.py
#### Description:
Predicts input image class using trained model checkpoint.

#### Options:
- image_path : location of input image
- checkpoint: loaction of trained model checkpoint
- -h, --help : predict.py options help
- --top_k : return top K most likely classes
- --category_names: use mapping of class indices to return real class names using .json dictionary
- --gpu : Use gpu for training, if available

#### Usage:
- basic usage:
`python predict.py '/img/test_img.jpg' 'checkpoints/checkpoint1.pth'`

- advanced usage:
`python predict.py '/img/test_img.jpg' 'checkpoints/checkpoint1.pth' --top_k 5 --category_names cat_to_names.json --gpu`

## Requirements
Python 3.6 + with the following packages:
- signal
- contextlib
- requests
- argparse
- numpy
- PIL
- json
- matplotlib
- torch
- torchvision
- time
- copy
- os
- sys

## Authors
- Alexander Manasson
- Udacity
