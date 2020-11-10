import argparse
import datetime
import json
import numpy as np
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append("..")

from training_loops import training_loop_UNet, training_loop_GAN
from networks.GANWrapper import GAN
from networks.ResidualNet import ResNet
from networks.UNet import UNet
from utils.DataLoader import ImgLoader


""" Training script """

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
arguments = parser.parse_args()

# Parse config json
with open(arguments.config_path, 'r') as infile:
    CONFIG = json.load(infile)

# Initialise datasets
TrainGenerator = ImgLoader(
    config=CONFIG,
    dataset_type="training",
    fold=0)

ValGenerator = ImgLoader(
    config=CONFIG,
    dataset_type="validation",
    fold=0)
# TODO: convert to have one generator for train and val
train_ds = tf.data.Dataset.from_generator(
    generator=TrainGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32)
    ).batch(CONFIG["HYPERPARAMS"]["MB_SIZE"])

val_ds = tf.data.Dataset.from_generator(
    generator=ValGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32)
    ).batch(CONFIG["HYPERPARAMS"]["MB_SIZE"])

# Compile model
# Model = UNet(nc=NC, lambda_=0.0, optimiser=keras.optimizers.Adam(ETA))
# Model = ResNet(nc=NC, optimiser=keras.optimizers.Adam(ETA))
Model = GAN(
    g_optimiser=keras.optimizers.Adam(2e-4, 0.5, 0.999),
    d_optimiser=keras.optimizers.Adam(2e-4, 0.5, 0.999),
    lambda_=100)

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



# Seg phase training loop
# training_loop_UNet(EPOCHS, "seg", Model, (train_ds, val_ds))

# Virtual contrast phase training loop
# training_loop_UNet(EPOCHS, "vc", Model, (train_ds, val_ds))

# GAN training loop
training_loop_GAN(CONFIG, Model, (train_ds, val_ds), False)
