# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import time
import argparse

#Parsing command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--testing_image", required = True, help = "Enter the path of testing image")
ap.add_argument("-model", "--model_directory", required = True, help = "Enter the path of model file")
args = vars(ap.parse_args())
print(args)

#Loading the image and testing
loaded_model = load_model(args["model_directory"])
loaded_model.summary()
img = image.load_img(args["testing_image"], target_size = (32,32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
stime = time.time()
prediction = loaded_model.predict_classes(x)
total = time.time() - stime
print("Time taken for inference {} s".format(total))