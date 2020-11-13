import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
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
    fold=5)

ValGenerator = ImgLoader(
    config=CONFIG,
    dataset_type="validation",
    fold=5)

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
    config=CONFIG,
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
results = training_loop_GAN(CONFIG, Model, (train_ds, val_ds), False)
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(results["epochs"], results["losses"]["G"], 'k', label="G")
plt.plot(results["epochs"], results["losses"]["D1"], 'r', label="D1")
plt.plot(results["epochs"], results["losses"]["D2"], 'g', label="D2")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Losses")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(results["epochs"], results["train_metrics"]["global"], 'k--', label="Train global L1")
plt.plot(results["epochs"], results["train_metrics"]["focal"], 'r--', label="Train focal L1")
plt.plot(results["epochs"], results["val_metrics"]["global"], 'k', label="Val global L1")
plt.plot(results["epochs"], results["val_metrics"]["focal"], 'r', label="Val focal L1")
plt.xlabel("Epochs")
plt.ylabel("L1")
plt.title("Metrics")
plt.legend()

plt.tight_layout()
plt.savefig(f"{CONFIG['SAVE_PATH']}logs/GAN/losses.png")

with open(f"{CONFIG['SAVE_PATH']}logs/GAN/results.json", 'w') as outfile:
    json.dump(results, outfile, indent=4)
