import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the dataset directory
dataset_dir = r'D:\ISEN\TP3\Dataset'

# Define the path to the output directories for training, validation, and test data
train_dir = r'D:\ISEN\TP3\train'
valid_dir = r'D:\ISEN\TP3\validation'
test_dir = r'D:\ISEN\TP3\test'

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
    train_images, valid_images = train_test_split(train_images, test_size=valid_split/(train_split+valid_split), random_state=42)

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

# Define the batch size for training and validation data
batch_size = 32

# Define the image augmentation settings for the training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Define the image augmentation settings for the validation and test data
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# Generate the validation data
valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                       target_size=img_size,
                                                       batch_size=batch_size,
                                                       class_mode='categorical')

# Generate the test data
test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

# Print the number of classes in the dataset
num_classes = len(train_generator.class_indices)
print('Number of classes:', num_classes)

# Print the size of the training, validation, and test sets
print('Training set size:', len(train_generator.filenames))
print('Validation set size:', len(valid_generator.filenames))
print('Test set size:', len(test_generator.filenames))

