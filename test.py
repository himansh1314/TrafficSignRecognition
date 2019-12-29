# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
import time
loaded_model = load_model('models/model.h5')
loaded_model.summary()
img = image.load_img('stop_test2.jpg', target_size = (32,32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
stime = time.time()
prediction = loaded_model.predict_classes(x)
total = time.time() - stime
print("Time taken for inference {} s".format(total))
