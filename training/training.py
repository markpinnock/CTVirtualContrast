import matplotlib.pyplot as plt
import numpy as np
# import os
import sys
import tensorflow.keras as keras
import tensorflow as tf
import time

sys.path.append('..')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep


SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/"
FILE_PATH = "Z:/Virtual_Contrast_Data/"

MB_SIZE = 4
EPOCHS = 10

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, MB_SIZE, True], output_types=tf.float32)

val_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, MB_SIZE, False], output_types=tf.float32)

Model = UNetGen(MB_SIZE)
print(Model.summary())

train_metric = keras.metrics.MeanSquaredError()
val_metric = keras.metrics.MeanSquaredError()
optimiser = keras.optimizers.Adam(1e-4)

start_time = time.time()

for epoch in range(EPOCHS):
    for imgs in train_ds:
        CE = imgs[0, :, :, :, :, tf.newaxis]
        NCE = imgs[1, :, :, :, :, tf.newaxis]
        trainStep(CE, NCE, Model, optimiser, train_metric)

    print("Epoch {}, Loss {}".format(epoch, train_metric.result()))

    train_metric.reset_states()

print(f"Time taken: {time.time() - start_time}")
count = 0

for imgs in train_ds:
    CE = imgs[0, :, :, :, :, tf.newaxis]
    NCE = imgs[1, :, :, :, :, tf.newaxis]
    pred = Model(NCE, training=False).numpy()
    # print(tf.reshape(thetas, [MB_SIZE, 3, 4]))
    CE = imgs[0, :, :, :, :, tf.newaxis].numpy()
    NCE = imgs[1, :, :, :, :, tf.newaxis].numpy()

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(np.flipud(CE[0, :, :, 0, 0]), cmap='gray', origin='lower', vmin=0.12, vmax=0.18)
    axs[0, 1].imshow(np.flipud(NCE[0, :, :, 0, 0]), cmap='gray', origin='lower', vmin=0.12, vmax=0.18)
    axs[0, 2].imshow(np.flipud(pred[0, :, :, 0, 0]), cmap='gray', origin='lower', vmin=0.12, vmax=0.18)
    axs[1, 0].imshow(np.flipud(pred[0, :, :, 0, 0] - CE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[1, 1].imshow(np.flipud(pred[0, :, :, 0, 0] - NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[1, 2].imshow(np.flipud(NCE[0, :, :, 0, 0] - CE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    plt.savefig(f"{SAVE_PATH}{count:03d}.png", dpi=250)
    plt.close()
    plt.show()
    count += 1
