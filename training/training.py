import matplotlib.pyplot as plt
# import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append('..')

from Networks import Network
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep


FILE_PATH = "C:/Users/rmappin/PhD_Data/Virtual_Contrast_Data/Train/"
FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/tf2_prac/train/"
MB_SIZE = 4
EPOCHS = 5

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, MB_SIZE, False], output_types=tf.float32)

val_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, MB_SIZE, False], output_types=tf.float32)

Model = Network()
loss_func = keras.losses.MeanSquaredError()
train_metric = keras.metrics.MeanSquaredError()
val_metric = keras.metrics.MeanSquaredError()
optimiser = keras.optimizers.Adam()

for epoch in range(EPOCHS):
    for imgs in train_ds:
        CE = imgs[0, :, :, :, :, tf.newaxis]
        NCE = imgs[1, :, :, :, :, tf.newaxis]
        trainStep(CE, NCE, Model, optimiser, loss_func, train_metric)

    print("Epoch {}, Loss {}".format(epoch, train_metric.result()))

    train_metric.reset_states()
