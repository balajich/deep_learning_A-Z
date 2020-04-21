#
# /!\ You first have to generate the model with the script:
# Volume_1-Supervised_Deep_Learning/Part_2-Convolutional_Neural_Networks-CNN/Section_10-Building_a_CNN/cnn.py
#
# from tensorflow.contrib.keras.api.keras.preprocessing import image
# from tensorflow.contrib.keras import backend
# from tensorflow.contrib.keras.api.keras.models import load_model
import os

import numpy as np
from keras.engine.saving import load_model
from keras.preprocessing import image

script_dir = os.path.dirname(__file__)
# Load pre-trained model
model_backup_path = os.path.join(script_dir, '../dataset/covid_or_normal_model.h5')
test_set_path = os.path.join(script_dir, '../dataset/c19/single_prediction')

classifier = load_model(model_backup_path)

input_size = (128, 128)

test_images_path = [test_set_path + '/' + filename for filename in os.listdir(test_set_path)]
test_images = np.array([image.img_to_array(image.load_img(test_image_name, target_size=input_size))
                        for test_image_name in test_images_path])

# No need to rescale the images here... why?

predictions = classifier.predict(test_images)

for prediction, image_path in zip(predictions, test_images_path):
    if prediction == 1:
        prediction = 'normal'
    else:
        prediction = 'covid'
    print("Predicted {} for file {}".format(prediction, image_path.split("/")[-1]))

# backend.clear_session()
