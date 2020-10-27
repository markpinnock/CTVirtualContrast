import datetime
import matplotlib.pyplot as plt
import numpy as np
# import os
import sys
import tensorflow.keras as keras
import tensorflow as tf
import time

sys.path.append('..')

from ResidualNet import ResNet
from UNet import UNet
from utils.DataLoader import ImgLoader


""" Training script """

SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/"
FILE_PATH = "C:/ProjectImages/VirtualContrast/"

# Hyperparameters
MB_SIZE = 4
NC = 4
EPOCHS = 100
ETA = 1e-4

# Initialise datasets
TrainGenerator = ImgLoader(
    file_path=FILE_PATH,
    dataset_type="training",
    num_folds=0,
    fold=0)

ValGenerator = ImgLoader(
    file_path=FILE_PATH,
    dataset_type="validation",
    num_folds=0,
    fold=0)
# TODO: convert to have one generator for train and val
train_ds = tf.data.Dataset.from_generator(
    generator=TrainGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32)
    ).batch(MB_SIZE)

val_ds = tf.data.Dataset.from_generator(
    generator=ValGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32)
    ).batch(MB_SIZE)

# Compile model
Model = UNet(nc=NC, lambda_=0.5, optimiser=keras.optimizers.Adam(ETA))
# Model = ResNet(nc=NC, optimiser=keras.optimizers.Adam(ETA))

# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/logs/" + curr_time
# writer = tf.summary.create_file_writer(log_dir)

# @tf.function
# def trace(x):
#     return Model(x)

# tf.summary.trace_on(graph=True)
# trace(tf.zeros((1, 128, 128, 12, 1)))
# # print(Model.summary())


# with writer.as_default():
#     tf.summary.trace_export('graph', step=0)
# exit()
start_time = time.time()

# Training loop
for epoch in range(EPOCHS):
    Model.metric.reset_states()

    for data in train_ds:
        Model.train_step(data)

    print(f"Epoch {epoch + 1}, Loss {Model.metric.result()}")


print(f"Time taken: {time.time() - start_time}")
count = 0

for data in val_ds:
    NCE, ACE, _ = data
    pred = Model(NCE, training=False).numpy()

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(np.flipud(NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[0, 1].imshow(np.flipud(ACE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[0, 2].imshow(np.flipud(pred[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[1, 0].imshow(np.flipud(pred[0, :, :, 0, 0] + NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[1, 1].imshow(np.flipud(pred[0, :, :, 0, 0] + NCE[0, :, :, 0, 0] - ACE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    axs[1, 2].imshow(np.flipud(pred[0, :, :, 0, 0] - ACE[0, :, :, 0, 0]), cmap='gray', origin='lower')
    # plt.savefig(f"{SAVE_PATH}{count:03d}.png", dpi=250)
    # plt.close()
    plt.show()
    count += 1
