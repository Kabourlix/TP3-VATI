import os

import matplotlib.pyplot as plt
import numpy as np

from keras.optimizers.optimizer_v1 import adam

from sklearn.metrics import make_scorer, confusion_matrix, cohen_kappa_score, classification_report

from pre_traitement import test_generator
from keras.models import load_model
import keras.utils as image

# Load image test

img_path = []
datatest_path = "D:/UQAC hiver 2023/8INF804-Vision artificielle/TP3-VATI/test"

# Load 5 images in each folder of the test dataset
for folder_name in os.listdir(datatest_path):
    folder_path = os.path.join(datatest_path, folder_name)
    if os.path.isdir(folder_path):
        images_to_test = 5
        images_tested = 0
        for file in os.listdir(folder_path):
            if images_tested < images_to_test:
                img_path.append(os.path.join(folder_path, file))
                images_tested += 1
            else:
                break

X_test = []
for img in img_path:
    img_test = image.load_img(img, target_size=(150, 150))
    img_arr = image.img_to_array(img_test) / 255.0
    X_test.append(img_arr)

X_test = np.array(X_test)
y_true = []
for i in range(len(X_test)):
    folder_name = os.path.basename(os.path.dirname(img_path[i]))
    y_true.append(folder_name)

# Define the custom_objects dictionary with the custom optimizer
custom_objects = {'custom_optimizer': adam}

# Load the model with the custom_objects argument
saved_model = load_model('vggclf.h5', custom_objects=custom_objects)

label_mapping = dict([(v, k) for k, v in test_generator.class_indices.items()])  # on associe chaque id à un label

# Make predictions
predictions = saved_model.predict(X_test)
y_pred = np.argmax(predictions, axis=-1)
y_pred_label = [label_mapping[int(prediction_id)] for prediction_id in y_pred]

# Compute the Cohen's kappa coefficient
kappa = cohen_kappa_score(y_true, y_pred_label)
# Print the kappa score
print("Cohen's kappa coefficient: ", kappa)

# Print the classification report and confusion matrix
print("Results of the test set:")
print(classification_report(y_true, y_pred_label))

# Show each image with the prediction
for i, img in enumerate(img_path):
    img_test = image.load_img(img, target_size=(150, 150))
    img_arr = image.img_to_array(img_test) / 255.0
    img_input = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))

    prediction_id = np.argmax(saved_model.predict(img_input), axis=-1)
    prediction_label = label_mapping[int(prediction_id)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_test)
    ax.set_title(prediction_label)
    ax.axis('off')
    plt.show()

