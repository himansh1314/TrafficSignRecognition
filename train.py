# -*- coding: utf-8 -*-
## Importing necessary libraries
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
## Initialise data generators for training and validation
train_gen = ImageDataGenerator(rescale = 1./255,
                               rotation_range = 10,
                               width_shift_range = 0.1,
                               height_shift_range = 0.1,
                               shear_range = 0.15,
                               horizontal_flip = False,
                               vertical_flip = False,
                               fill_mode = "nearest",
                               zoom_range = 0.15
                               )
validation_gen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 10,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    shear_range = 0.15,
                                    horizontal_flip = False,
                                    vertical_flip = False,
                                    fill_mode = "nearest",
                                    zoom_range = 0.15
                                    )

## creating the model
model = Sequential([
    Conv2D(6, (5,5), activation = 'relu', input_shape = (32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(16,(5,5), activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(120, activation = 'relu'),
    Dropout(0.2),
    Dense(84, activation = 'relu'),
    Dropout(0.2),
    Dense(43, activation= 'softmax')
    ])
model.compile(optimizer = RMSprop(lr=0.001), loss = 'categorical_crossentropy', metrics = ['acc'])
model.summary()

## directories
TRAIN_DIRECTORY = 'dataset/train' 
VAL_DIRECTORY = 'dataset/val'

## data flow from generator
train_generator = train_gen.flow_from_directory(TRAIN_DIRECTORY, target_size = (32,32), class_mode='categorical')
validation_generator = validation_gen.flow_from_directory(VAL_DIRECTORY, target_size = (32,32), class_mode='categorical')

## Train the model
history = model.fit_generator(train_generator, epochs = 25, validation_data = validation_generator, verbose = 1)

## Save the trained model
model.save('models/model.h5')

## Plot history of training and validation accuracy and loss
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs = range(len(acc))
## Accuracy
plt.plot(epochs,acc,'r', "Training accuracy")
plt.plot(epochs, val_acc, 'b', "Validation accuracy")
plt.title('Training and Validation Accuracy')
plt.figure()

##Loss
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")

