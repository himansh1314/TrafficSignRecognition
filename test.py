# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import time
import argparse

#Parsing command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--testing_image", required = True, help = "Enter the path of testing image")
ap.add_argument("-model", "--model_directory", required = True, help = "Enter the path of model file")
args = vars(ap.parse_args())
print(args)

corrected_label = {
0: "speed limit 20", 
1: "speed limit 30", 
12: "speed limit 50", 
23: "speed limit 60", 
34 : "speed limit 70", 
39 : "speed limit 80", 
40 : "restriction ends 80", 
41: "speed limit 100", 
42: "speed limit 120", 
43: "no overtaking", 
2 : "no overtaking (trucks)", 
3 : "priority at next intersection", 
4 : "priority road",
5 : "give way", 
6 : "stop", 
7 : "no traffic both ways", 
8 : "no trucks", 
9 : "no entry", 
10 : "danger", 
11 : "bend left", 
13 : "bend right", 
14 : "bend",
15 : "uneven road", 
16 : "slippery road", 
17 : "road narrows", 
18 : "construction", 
19 : "traffic signal", 
20 : "pedestrian crossing", 
21 : "school crossing", 
22 : "cycles crossing", 
24 : "snow", 
25 : "animals", 
26 : "restriction ends", 
27 : "go right", 
28 : "go left", 
29 : "go straight", 
30 : "go right or straight", 
31 : "go left or straight", 
32 : "keep right", 
33 : "keep left", 
35 : "roundabout", 
37 : "restriction ends (overtaking)", 
38 : "restriction ends (overtaking (trucks))",
}
#Loading the image and testing
loaded_model = load_model(args["model_directory"])
loaded_model.summary()
img = image.load_img(args["testing_image"], target_size = (32,32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
stime = time.time()
prediction = loaded_model.predict_classes(x)
print(corrected_label[int(prediction)])
total = time.time() - stime
print("Time taken for inference {} s".format(total))
