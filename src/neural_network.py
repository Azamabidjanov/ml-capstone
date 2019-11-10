import tensorflow as tf
from tensorflow import keras
#from keras.datasets import cifar10, mnist
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from pathlib import Path
from fastai.vision import *
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os

# path for data folders
path = Path(os.getcwd())/"data"

# load data using fastai method
tfms = get_transforms(do_flip=True,flip_vert=True)
data = ImageDataBunch.from_folder(path,test="test",ds_tfms=tfms,bs=16)

# show data and labels
#data.show_batch(rows=4,figsize=(10,8))
#plt.show()

# CNN model
learn = create_cnn(data,models.resnet34,metrics=error_rate)

learn.save("keras-recycle")

# Show error
#learn.lr_find(start_lr=1e-6,end_lr=1e1)
#learn.recorder.plot()
#plt.show()

#learn.fit_one_cycle(20,max_lr=5.13e-03) ->run 20 epoches, at 8.6%, validation errors looks best







#img = cv2.imread('..\dataset_resized\dataset_resized\trash\trash2.jpg')
#cv2.imshow('Image', img)
#cv2.waitKey()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#example = x_train[0]

#plt.imshow(example, cmap="gray", vmin=0, vmax=255)
#plt.show()
