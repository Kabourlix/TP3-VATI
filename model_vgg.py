from tensorflow import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

# Architecture VGG16

# Formatter image data
tr_data = ImageDataGenerator(rescale=1./255)
train_data = tr_data.flow_from_directory(directory="./train",target_size=(150,150))
vd_data = ImageDataGenerator(rescale=1./255)
validation_data = vd_data.flow_from_directory(directory="./validation",target_size=(150,150))

# VGG : modèle pré-entrainé
VGG = keras.applications.VGG16(input_shape=(150,150,3),include_top=False,weights='imagenet')
VGG.trainable = False

# Construire notre VGG
model = Flatten()(VGG.output)
model = Dense(512,activation="relu")(model)
model = Dropout(0.2)(model)
model = Dense(512,activation="relu")(model)
model = Dropout(0.2)(model)
model = Dense(53,activation="softmax")(model)

# Créer modèle et afficher summary
model = Model(inputs=VGG.input, outputs=model)
model.summary()

# Compiler modèle
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit
early_stop=keras.callbacks.EarlyStopping(patience=5)
hist = model.fit(train_data,epochs=15,validation_data=validation_data,verbose=1,validation_steps=3,callbacks=early_stop)

# Sauvegarder modèle
model.save('./vggV2.h5')

# Visualisation
plt.style.use('ggplot')
plt.figure()
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
plt.show()
plt.savefig("./plotVGG_training.png")
