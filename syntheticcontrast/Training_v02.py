import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras as keras
import tensorflow as tf

from trainingtuningclasses.TrainingClasses_v02 import TrainingLoopUNet, TrainingLoopGAN
from networks.GANWrapper import GAN, CropGAN_v01
from networks.UNet import UNet, CropUNet
from utils.DataLoader import PairedLoader, UnpairedLoader


""" Training entry point """

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
parser.add_argument("--expt_name", "-en", help="Expt name", type=str)
parser.add_argument("--save_every", "-s", help="Save every _ epochs", type=int)
parser.add_argument("--lambda_", "-l", help="Lambda", type=float)
parser.add_argument("--gpu", "-g", help="GPU number", type=int)
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

# Set GPU
if arguments.gpu is not None:
    gpu_number = arguments.gpu
    os.environ["LD_LIBRARY_PATH"] = CONFIG["CUDA_PATH"]
else:
    gpu_number = 0

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[gpu_number], "GPU")

if CONFIG["DATA"]["DATA_TYPE"] == "paired":
    Loader = PairedLoader

elif CONFIG["DATA"]["DATA_TYPE"] == "unpaired":
    Loader = UnpairedLoader

else:
    raise ValueError("Select paired or unpaired dataloader")

# Initialise datasets
if CONFIG["DATA"]["NORM_PARAM_1"] == "" or CONFIG["DATA"]["NORM_PARAM_2"] == "":
    param_1 = None
    param_2 = None

else:
    param_1 = CONFIG["DATA"]["NORM_PARAM_1"]
    param_2 = CONFIG["DATA"]["NORM_PARAM_2"]

TrainGenerator = Loader(config=CONFIG, dataset_type="training")
param_1, param_2 = TrainGenerator.set_normalisation(CONFIG["DATA"]["NORM_TYPE"], param_1, param_2)
ValGenerator = Loader(config=CONFIG, dataset_type="validation")
_, _ = ValGenerator.set_normalisation(CONFIG["DATA"]["NORM_TYPE"], param_1, param_2)

CONFIG["DATA"]["NORM_PARAM_1"] = int(param_1)
CONFIG["DATA"]["NORM_PARAM_2"] = int(param_2)

# Batch size (separate batches for generator and critic runs)
if CONFIG["EXPT"]["MODEL"] == "GAN":
    MB_SIZE = CONFIG["HYPERPARAMS"]["MB_SIZE"] + CONFIG["HYPERPARAMS"]["MB_SIZE"] * CONFIG["HYPERPARAMS"]["N_CRITIC"]
else:
    MB_SIZE = CONFIG["HYPERPARAMS"]["MB_SIZE"]

output_tensors = (tf.float32, tf.float32)

# Create dataloader
train_ds = tf.data.Dataset.from_generator(
    generator=TrainGenerator.data_generator,
    output_types=output_tensors
    ).batch(MB_SIZE)

val_ds = tf.data.Dataset.from_generator(
    generator=ValGenerator.data_generator,
    output_types=output_tensors
    ).batch(MB_SIZE)

# Compile model
Model = GAN(config=CONFIG)
d_opt = keras.optimizers.Adam(CONFIG["HYPERPARAMS"]["D_ETA"], 0.5, 0.999, name="d_opt")
g_opt = keras.optimizers.Adam(CONFIG["HYPERPARAMS"]["G_ETA"], 0.5, 0.999, name="g_opt")

Model.compile(g_optimiser=g_opt, d_optimiser=d_opt, loss="minmax")

if CONFIG["EXPT"]["VERBOSE"]:
    Model.summary()

if CONFIG["EXPT"]["GRAPH"]:
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/logs/" + CONFIG["EXPT"]["MODEL"] + "/" + curr_time
    writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def trace(x):
        return Model.Generator(x)

    tf.summary.trace_on(graph=True)
    trace(tf.zeros((1, 128, 128, 12, 1)))

    with writer.as_default():
        tf.summary.trace_export('graph', step=0)
    exit()

TrainingLoop = TrainingLoopGAN(Model=Model, dataset=(train_ds, val_ds), config=CONFIG)

# Run training loop
TrainingLoop.training_loop()
TrainingLoop.save_results()
