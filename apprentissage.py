from tensorflow import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

# Architecture VGG16

# Formatter image data
tr_data = ImageDataGenerator(rescale=1./255)
train_data = tr_data.flow_from_directory(directory=r'C:\Users\annie\Documents\GitHub\TP3-VATI\train',target_size=(150,150))
vd_data = ImageDataGenerator(rescale=1./255)
validation_data = vd_data.flow_from_directory(directory=r'C:\Users\annie\Documents\GitHub\TP3-VATI\validation',target_size=(150,150))

# VGG
VGG = keras.applications.VGG16(input_shape=(150,150,3),include_top=False,weights='imagenet')
# Les layers sont présents à l'intérieur de cette variable VGG. VGG est inbuilt
VGG.trainable = False # On ne veut pas train les premiers 13 layers mais seulement les 2 derniers

# On aura 1x Dense layer de 4096 units, 1x Dense layer de 4096 units et 1x Dense Softmax layer de 2 units

# Création modèle et compilation
model = keras.Sequential([
    VGG,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=53,activation="softmax")
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Summary/Conclusion
model.summary()
early_stop=keras.callbacks.EarlyStopping(patience=5)
hist = model.fit(train_data,epochs=5,validation_data=validation_data,verbose=1,validation_steps=3,callbacks=early_stop)
model.save('./vggclf.h5') #Sauvegarder modèle

# Visualisation
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
plt.show
