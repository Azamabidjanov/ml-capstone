import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10, mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from pathlib import Path
from fastai.vision import *
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

path = Path(os.getcwd())/"data"

data = ImageDataBunch.from_folder(path,test="test",ds_tfms=tfms,bs=16)

#img = cv2.imread('..\dataset_resized\dataset_resized\trash\trash2.jpg')
#cv2.imshow('Image', img)
#cv2.waitKey()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#example = x_train[0]

#plt.imshow(example, cmap="gray", vmin=0, vmax=255)
#plt.show()
