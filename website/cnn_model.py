import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import numpy as np
import scipy
from PIL import Image
import glob
#import matplotlib
#import matplotlib.pyplot as plt
import re

def get_train_data():
    # Load and convert to grayscale
    img_list = []
    img_labels = []
    for directory in glob.glob('../data/train/*/'):
        for filename in glob.glob(directory + '*.jpg'):
            img = np.asarray(Image.open(filename).convert('LA'))
            img_list.append(img)
            
            # Find label
            pic_type = re.search("cardboard|glass|metal|paper|plastic|trash", directory).group()
            if pic_type == 'cardboard':
                img_labels.append(0)
            elif pic_type == 'glass':
                img_labels.append(1)
            elif pic_type == 'metal':
                img_labels.append(2)
            elif pic_type == 'paper':
                img_labels.append(3)
            elif pic_type == 'plastic':
                img_labels.append(4)
            else:
                img_labels.append(5)
    
    print(str(len(img_list)) + ' images converted.')
    return np.asarray(img_list), np.asarray(img_labels)

def get_test_data():
    # Load and convert to grayscale
    img_list = []
    img_labels = []
    for filename in glob.glob('../data/test/*.jpg'):
        img = np.asarray(Image.open(filename).convert('LA'))
        img_list.append(img)
            
        # Find label
        pic_type = re.search("cardboard|glass|metal|paper|plastic|trash", filename).group()
        if pic_type == 'cardboard':
            img_labels.append(0)
        elif pic_type == 'glass':
            img_labels.append(1)
        elif pic_type == 'metal':
            img_labels.append(2)
        elif pic_type == 'paper':
            img_labels.append(3)
        elif pic_type == 'plastic':
            img_labels.append(4)
        else:
            img_labels.append(5)
    
    print(str(len(img_list)) + ' images converted.')
    return np.asarray(img_list), np.asarray(img_labels)

def get_model(x_train, y_train):
    # Sizing
    img_rows, img_cols = 384, 512
    num_classes = 6
    input_shape = (img_rows, img_cols, 1)
    
    # Model time - CNN
    layers = [
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
        ]
    
    # Train and return
    cnn_model = Sequential(layers)
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.fit(x_train, y_train, epochs=3, validation_split=0.2)
    print('Model compiled successfully!')
    return cnn_model

def main():
    # Load images and labels
    train_images,train_labels = get_train_data()
    test_images,test_labels = get_test_data()
    x_train = train_images.astype('float32')
    y_train = train_labels.astype('float32')
    x_test = test_images.astype('float32')
    y_test = test_labels.astype('float32')

    # Create our model
    model = get_model(x_train, y_train)
    
    # Test that bitch
    print('Testing goes here')

if __name__ == '__main__':
    main()
