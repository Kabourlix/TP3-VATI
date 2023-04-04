import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import time
import matplotlib.pyplot as plt
from keras.optimizers.optimizer_v1 import adam
from sklearn.metrics import cohen_kappa_score, classification_report
from pre_traitement import test_generator
from keras.models import load_model
import keras.utils as image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

''' Division du dossier DataSet en trois sous-ensembles  '''

# Récupérer le chemin vers les dossiers
dataset_dir = r'Dataset'
train_dir = r'train'
valid_dir = r'validation'
test_dir = r'test'

# Create the output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratio for training, validation, and test data
train_split = 0.7
valid_split = 0.15
test_split = 0.15

# Define the image size for resizing
img_size = (224, 224)

# Loop through each class directory in the dataset directory
for class_dir in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_dir)
    if not os.path.isdir(class_path):
        continue

    # Get a list of all image files in the class directory
    images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg')]

    # Split the images into training, validation, and test sets
    train_images, test_images = train_test_split(images, test_size=test_split, random_state=42)
    train_images, valid_images = train_test_split(train_images, test_size=valid_split / (train_split + valid_split),
                                                  random_state=42)

    # Create subdirectories for the class in the training, validation, and test directories
    train_class_dir = os.path.join(train_dir, class_dir)
    valid_class_dir = os.path.join(valid_dir, class_dir)
    test_class_dir = os.path.join(test_dir, class_dir)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(valid_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Resize and move the images into the appropriate subdirectories
    for image in train_images:
        img = Image.open(image)
        img = img.resize(img_size)
        dest_path = os.path.join(train_class_dir, os.path.basename(image))
        img.save(dest_path)
    for image in valid_images:
        img = Image.open(image)
        img = img.resize(img_size)
        dest_path = os.path.join(valid_class_dir, os.path.basename(image))
        img.save(dest_path)
    for image in test_images:
        img = Image.open(image)
        img = img.resize(img_size)
        dest_path = os.path.join(test_class_dir, os.path.basename(image))
        img.save(dest_path)

''' Pas d'entrainement des modèles ici!'''
#  voir "model_vgg.py" ou "model_cnn.py"

''' Test et analyse des résultats'''

''' Pour VGG : '''

# Load image test
img_path = []
datatest_path = r"test"

# Load all images in each folder of the test dataset
for folder_name in os.listdir(datatest_path):
    folder_path = os.path.join(datatest_path, folder_name)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            img_path.append(os.path.join(folder_path, file))

X_test = []
for img1 in img_path:
    img_test = load_img(img1, target_size=(150, 150))
    img_arr = img_to_array(img_test) / 255.0
    X_test.append(img_arr)

X_test = np.array(X_test)
y_true = []
for i in range(len(X_test)):
    folder_name = os.path.basename(os.path.dirname(img_path[i]))
    y_true.append(folder_name)

# Define the custom_objects dictionary with the custom optimizer
custom_objects = {'custom_optimizer': adam}

# Load the model with the custom_objects argument
saved_model = load_model('vggV2.h5', custom_objects=custom_objects)

label_mapping = dict([(v, k) for k, v in test_generator.class_indices.items()])  # on associe chaque id à un label

# Make predictions
predictions = saved_model.predict(X_test)
y_pred = np.argmax(predictions, axis=-1)
y_pred_label = [label_mapping[int(prediction_id)] for prediction_id in y_pred]

print('VGG16 architecture performance results:')
print()
# Compute the Cohen's kappa coefficient
kappa = cohen_kappa_score(y_true, y_pred_label)
# Print the kappa score
print("Cohen's kappa coefficient for the VGG16 architecture: ", kappa)
print()

# Print the classification report
print("Classification report of the VGG16 architecture:")
print(classification_report(y_true, y_pred_label))

# Show each image with the prediction
for i, img in enumerate(img_path):
    img_test = load_img(img, target_size=(150, 150))
    img_arr = img_to_array(img_test) / 255.0
    img_input = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))

    prediction_id = np.argmax(saved_model.predict(img_input), axis=-1)
    prediction_label = label_mapping[int(prediction_id)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_test)
    ax.set_title(prediction_label)
    ax.axis('off')
    plt.show()

''' Pour CNN : '''

# Load image test

img_path = []
datatest_path = r"test"

# Load all images in each folder of the test dataset
for folder_name in os.listdir(datatest_path):
    folder_path = os.path.join(datatest_path, folder_name)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            img_path.append(os.path.join(folder_path, file))

X_test = []
for img2 in img_path:
    img_test = load_img(img2, target_size=(150, 150))
    img_arr = img_to_array(img_test) / 255.0
    X_test.append(img_arr)

X_test = np.array(X_test)
y_true = []
for i in range(len(X_test)):
    folder_name = os.path.basename(os.path.dirname(img_path[i]))
    y_true.append(folder_name)

# Define the custom_objects dictionary with the custom optimizer
custom_objects = {'custom_optimizer': adam}

# Load the model with the custom_objects argument
saved_model = load_model('cnnV1.h5', custom_objects=custom_objects)

label_mapping = dict([(v, k) for k, v in test_generator.class_indices.items()])  # on associe chaque id à un label

# Make predictions
predictions = saved_model.predict(X_test)
y_pred = np.argmax(predictions, axis=-1)
y_pred_label = [label_mapping[int(prediction_id)] for prediction_id in y_pred]

print('CNN architecture performance results:')
print()
# Compute the Cohen's kappa coefficient
kappa = cohen_kappa_score(y_true, y_pred_label)
# Print the kappa score
print("Cohen's kappa coefficient for the CNN architecture: ", kappa)
print()

# Print the classification report
print("Classification report of the CNN architecture:")
print(classification_report(y_true, y_pred_label))

# Show each image with the prediction
for i, img in enumerate(img_path):
    img_test = load_img(img, target_size=(150, 150))
    img_arr = img_to_array(img_test) / 255.0
    img_input = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))

    prediction_id = np.argmax(saved_model.predict(img_input), axis=-1)
    prediction_label = label_mapping[int(prediction_id)]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_test)
    ax.set_title(prediction_label)
    ax.axis('off')
    plt.show()