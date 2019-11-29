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

# It is necessary to call a method from main on Windows 10 with PyTorch
# ref: https://github.com/pytorch/pytorch/issues/5858
def run():

    # path for data folders
    path = Path(os.getcwd())/"data"

    print(path)

    # Augment data. Compare results by flipping images horizontally and vertically
    tfms = get_transforms(do_flip=True,flip_vert=True)

    # num_workers = 0 part is necessary on Windows to avoid broken pipe error
    data = ImageDataBunch.from_folder(path,test="test",ds_tfms=tfms,bs=16)

    # show data and labels
    # data.show_batch(rows=4,figsize=(10,8))
    # plt.show()

    # CNN model using ReseNet34
    learn = cnn_learner(data,models.resnet34,metrics=error_rate)

    # Show error
    #learn.lr_find(start_lr=1e-6,end_lr=1e1)
    #learn.recorder.plot()
    #plt.show()

    # Run on 20 epochs
    learn.fit_one_cycle(20,max_lr=5.13e-03) #->run 20 epoches, at 8.6%, validation errors looks best
    # Export to pkl file
    learn.export('recycle-fastai.pkl')


    # Visualizing most incorrect images
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    interp.plot_top_losses(9, figsize=(15,11))
    plt.show()

    # show confusion matrix
    doc(interp.plot_top_losses)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.show()

    # List of most confused images
    interp.most_confused(min_val=2)


if __name__ == '__main__':
    run()
