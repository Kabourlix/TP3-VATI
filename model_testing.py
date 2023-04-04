from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import keras.utils as image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

test_data = ImageDataGenerator(rescale=1./255)
validation_data = test_data.flow_from_directory(directory=r'C:\Users\annie\Documents\GitHub\TP3-VATI\test',target_size=(150,150))
label_mapping = dict([(v, k) for k, v in validation_data.class_indices.items()]) # on associe chaque id à un label

# Load image test
test_image_path = 'C:/Users/annie/Documents/GitHub/TP3-VATI/test/'
img_test = image.load_img('C:/Users/annie/Documents/GitHub/TP3-VATI/test/four of clubs/010.jpg',target_size=(150,150)) # changer l'image ici
img_arr = image.img_to_array(img_test)/255.0
img_input = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))

# Charger model
saved_model = load_model('./vggv2.h5')
saved_model.summary()

# on ne peut pas utiliser decode predictions car ce n'est pas un pre-trained model
# p = decode_predictions(features)

# Prédiction
prediction_id = saved_model.predict(img_input)
prediction_id = np.argmax(prediction_id,axis=1)
prediction_label = label_mapping[int(prediction_id)]

print(prediction_label)