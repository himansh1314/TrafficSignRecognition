# TrafficSignRecognition
Traffic sign recognition using CNN on GTSRB dataset
Designed a LeNet-5 architecture on Tensorflow2 and used the GTSRB dataset for training the model on over 40000 images across 43 classes.
# Requirements
- tensorflow-gpu
- matplotlib
# Files included
> train.py - Used to train the model. It provides a number of options as command line arguments. Example
"""
python train.py --epochs 100 --train_directory train_path --test_directory test_path
"""
This will train the model for 100 epochs and use the training data from train_path and testing data from test_path
> test.py - Used to test model on images.

# Results
Lenet is a simple and compact model with just under 65,000 parameters which results to an accuracy of over 95% on GTSRB dataset.
