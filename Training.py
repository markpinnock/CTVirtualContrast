import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras as keras
import tensorflow as tf

from TrainingLoops import TrainingLoopUNet, TrainingLoopGAN, print_model_summary
from networks.GANWrapper import BaseGAN, CropGAN
from networks.ResidualNet import ResNet
from networks.UNet import UNet
from utils.DataLoader import OneToOneLoader


""" Training script """

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
parser.add_argument("--expt_name", "-en", help="Expt name", type=str)
parser.add_argument("--save_every", "-s", help="Save every _ epochs", type=int)
parser.add_argument("--lambda_", "-l", help="Lambda", type=float)
parser.add_argument("--mu", "-m", help="Mu", type=float)
parser.add_argument("--d_layers_g", "-dgl", help="Global discriminator layers", type=int)
parser.add_argument("--d_layers_f", "-dfl", help="Focal discriminator layers", type=int)
parser.add_argument("--g_layers", "-gl", help="Generator layers", type=int)
arguments = parser.parse_args()

# Parse config json
with open(arguments.config_path, 'r') as infile:
    CONFIG = json.load(infile)

if arguments.expt_name is not None:
    CONFIG["EXPT"]["EXPT_NAME"] = arguments.expt_name
else:
    CONFIG["EXPT"]["EXPT_NAME"] = "test"

if arguments.save_every is not None:
    CONFIG["EXPT"]["SAVE_EVERY"] = arguments.save_every
else:
    CONFIG["EXPT"]["SAVE_EVERY"] = 1

if arguments.lambda_ is not None:
    CONFIG["HYPERPARAMS"]["LAMBDA"] = arguments.lambda_

if arguments.mu is not None:
    CONFIG["HYPERPARAMS"]["MU"] = arguments.mu

if arguments.d_layers_g is not None:
    CONFIG["HYPERPARAMS"]["D_LAYERS_G"] = arguments.d_layers_g

if arguments.d_layers_f is not None:
    CONFIG["HYPERPARAMS"]["D_LAYERS_F"] = arguments.d_layers_f

if arguments.g_layers is not None:
    CONFIG["HYPERPARAMS"]["G_LAYERS"] = arguments.g_layers

# Initialise datasets
TrainGenerator = OneToOneLoader(
    config=CONFIG["EXPT"],
    dataset_type="training",
    fold=5)

ValGenerator = OneToOneLoader(
    config=CONFIG["EXPT"],
    dataset_type="validation",
    fold=5)

# Batch size (separate batches for generator and critic runs)
if CONFIG["EXPT"]["MODEL"] == "GAN":
    MB_SIZE = CONFIG["HYPERPARAMS"]["MB_SIZE"] + CONFIG["HYPERPARAMS"]["MB_SIZE"] * CONFIG["HYPERPARAMS"]["N_CRITIC"]
else:
    MB_SIZE = CONFIG["HYPERPARAMS"]["MB_SIZE"]

# Create dataloader
train_ds = tf.data.Dataset.from_generator(
    generator=TrainGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
    ).batch(MB_SIZE)

val_ds = tf.data.Dataset.from_generator(
    generator=ValGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
    ).batch(MB_SIZE)

# Compile model
if CONFIG["EXPT"]["MODEL"] == "UNet":
    Model = UNet(config=CONFIG)
    TrainingLoop = TrainingLoopUNet(Model=Model, dataset=(train_ds, val_ds), config=CONFIG)

elif CONFIG["EXPT"]["MODEL"] == "GAN":
    if not CONFIG["EXPT"]["CROP"]:
        Model = BaseGAN(config=CONFIG)
    elif CONFIG["EXPT"]["CROP"]:
        Model = CropGAN(config=CONFIG)

    TrainingLoop = TrainingLoopGAN(Model=Model, dataset=(train_ds, val_ds), config=CONFIG)

else:
    raise ValueError

# if CONFIG["EXPT"]["VERBOSE"] and not CONFIG["HYPERPARAMS"]["D_LAYERS_F"]:
#     print_model_summary(Model.Generator, CONFIG, "G")
#     print_model_summary(Model.Discriminator, CONFIG, "D")
# elif CONFIG["EXPT"]["VERBOSE"] and CONFIG["HYPERPARAMS"]["D_LAYERS_F"]:
#     print_model_summary(Model.Generator, CONFIG, "G")

#     for name, Discriminator in Model.Discriminators.items():
#         print_model_summary(Discriminator, CONFIG, "D")
# else:
#     pass

# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/logs/" + curr_time
# writer = tf.summary.create_file_writer(log_dir)

# @tf.function
# def trace(x):
#     return Model.Generator(x)

# tf.summary.trace_on(graph=True)
# trace(tf.zeros((1, 128, 128, 12, 1)))

# with writer.as_default():
#     tf.summary.trace_export('graph', step=0)
# exit()

# Run training loop
TrainingLoop.training_loop()
TrainingLoop.save_results()

# with writer.as_default():
#     tf.summary.trace_export("graph", step=0)
