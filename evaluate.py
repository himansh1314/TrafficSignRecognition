# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

model = load_model('models/model.h5')

test_path = 'dataset/test'

test_gen = ImageDataGenerator(
    rescale = 1./255
    )

data_generator = test_gen.flow_from_directory(test_path, class_mode = 'categorical', target_size=(32,32), shuffle = False)

def load_preprocess(path):
    img = image.load_img(path, target_size=(32,32))
    img = image.img_to_array(img)   
    img = np.expand_dims(img, axis = 0)
    img = test_gen.standardize(img)
    return img

# prediction = model.predict(load_preprocess('IEEEDataport/16660_2_1.jpg'))
# print(prediction)

predictions = model.predict_generator(data_generator)
predictions = np.argmax(predictions, axis = 1)

class_label = [
"speed limit 20", 
"speed limit 30", 
"speed limit 50", 
"speed limit 60", 
"speed limit 70", 
"speed limit 80", 
"restriction ends 80", 
"speed limit 100", 
"speed limit 120", 
"no overtaking", 
"no overtaking (trucks)", 
"priority at next intersection", 
"priority road",
"give way", 
"stop", 
"no traffic both ways", 
"no trucks", 
"no entry", 
"danger", 
"bend left", 
"bend right", 
"bend",
"uneven road", 
"slippery road", 
"road narrows", 
"construction", 
"traffic signal", 
"pedestrian crossing", 
"school crossing", 
"cycles crossing", 
"snow", 
"animals", 
"restriction ends", 
"go right", 
"go left", 
"go straight", 
"go right or straight", 
"go left or straight", 
"keep right", 
"keep left", 
"roundabout", 
"restriction ends (overtaking)", 
"restriction ends (overtaking (trucks))",
]
##Confusion matrix and classification report
print(confusion_matrix(data_generator.classes, predictions))
print(classification_report(data_generator.classes, predictions, target_names = class_label))

