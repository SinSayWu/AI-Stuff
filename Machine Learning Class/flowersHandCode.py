import numpy as np
import os
import PIL
import PIL.Image
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt
from singleHiddenNNClass import neuralNetwork1Hidden as nn1
from multiHiddenNNClass import neuralNetworkMultiHidden as nnM

dataDir = "flower_photos"
def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(subdir)
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, target_size)  # Resize the image
                    img_flatten = img_resized.flatten()  # Flatten the resized image to a 1D array
                    images.append((img_flatten, label))
    return images

def split_dataset(images, train_size=500, test_size=100):
    random.shuffle(images)
    train_set = images[:train_size]
    test_set = images[train_size:train_size + test_size]
    return train_set, test_set

def split_dataset(images, train_size=500, test_size=100):
    random.shuffle(images)
    train_set = images[:train_size]
    test_set = images[train_size:train_size + test_size]
    return train_set, test_set
    
def save_to_csv(data, output_csv):
    images, labels = zip(*data)
    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")
    
folder_path = 'flower_photos'
train_csv = 'train_set.csv'
test_csv = 'test_set.csv'
target_size = (64, 64) # Size Specification

# Load all images
images = load_images_from_folder(folder_path, target_size)

# Split the dataset into training and testing sets
train_set, test_set = split_dataset(images, train_size=500, test_size=100)

# Save the training and testing sets to separate CSV files
save_to_csv(train_set, train_csv)
save_to_csv(test_set, test_csv)

data_file = open("train_set.csv",'r')
data_list =  data_file.readlines()
data_file.close()

all_values = data_list[1].split(',')
image_array =  np.asfarray(all_values[:-1]).reshape((64,64,3))
image_array_norm = image_array / 255

plt.imshow(image_array_norm)