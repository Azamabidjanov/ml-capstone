import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
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
            img = np.array(Image.open(filename).convert('L'))
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
    #print(np.asarray(img_list).shape)
    return np.asarray(img_list), np.asarray(img_labels)

def get_test_data():
    # Load and convert to grayscale
    img_list = []
    img_labels = []
    for filename in glob.glob('../data/test/*.jpg'):
        img = np.asarray(Image.open(filename).convert('L'))
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
    
    print(str(len(img_labels)) + ' images converted.')
    return np.asarray(img_list), np.asarray(img_labels)

def get_model(x_train, y_train):
    # Sizing
    img_rows, img_cols = 384, 512
    num_classes = 6
    input_shape = (img_rows, img_cols, 1)
    
    # Model time - CNN
    layers = [
        Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape),
        MaxPool2D(),
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
        ]
    
    # Train and return
    cnn_model = Sequential(layers)
    cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.fit(x_train.reshape(-1, img_rows, img_cols, 1), y_train, epochs=5, validation_split=0.2)
    print('Model trained successfully!')
    return cnn_model

def main():
    # Load images and labels
    train_images,train_labels = get_train_data()
    test_images,test_labels = get_test_data()
    x_train = train_images.astype('float32')
    y_train = train_labels.astype('float32')
    x_test = test_images.astype('float32')
    y_test = test_labels.astype('float32')
    
    # TESTING
    print(test_images.shape)
    print(x_test.shape)

    # Create our model
    model = get_model(x_train, y_train)
    
    # Test that bitch
    cnn_scores = model.evaluate(x_test.reshape(-1, 384, 512, 1), y_test)
    print('Accuracy of: ' + str(cnn_scores[1]))

if __name__ == '__main__':
    main()

