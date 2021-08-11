import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from trainingloops import TrainingLoopUNet, TrainingLoopGAN
from tuners.tuningclasses import GridSearch, RandomSearch

np.set_printoptions(suppress=True)


""" Hyper-parameter tuning script """

# Load configuration json
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", '-cp', help="Path to config json", type=str)
parser.add_argument("--tuning_path", '-tp', help="Path to tuning config json", type=str)
parser.add_argument("--algo", '-a', help="Tuning algorithm", type=str)
parser.add_argument("--expt_name", '-e', help="Experiment name", type=str)
parser.add_argument("--runs", '-r', help="Number of validation runs", type=int)
parser.add_argument("--gpu", "-g", help="GPU number", type=int)
args = parser.parse_args()

CONFIG = json.load(open(args.config_path, 'r'))
tuning_config = json.load(open(args.tuning_path, 'r'))

# Set number of runs and run algorithm
if args.runs is not None:
    RUNS = args.runs
else:
    RUNS = 100

if args.expt_name is not None:
    EXPT_NAME = args.expt_name
else:
    EXPT_NAME = "test"

# Set GPU
if args.gpu is not None:
    gpu_number = args.gpu
    print("==================================================")
    print(f"Using GPU {gpu_number}")
    os.environ["LD_LIBRARY_PATH"] = CONFIG["CUDA_PATH"]
else:
    gpu_number = 0
    print("==================================================")
    print(f"Using default GPU")

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[gpu_number], "GPU")

if CONFIG["EXPT"]["MODEL"] == "UNet":
    TrainingLoop = TrainingLoopUNet
elif CONFIG["EXPT"]["MODEL"] == "GAN":
    TrainingLoop = TrainingLoopGAN
else:
    raise ValueError("Model not recognised")

if args.algo == "grid":
    Grid = GridSearch(EXPT_NAME, CONFIG, tuning_config, TrainingLoop)
    Grid.tuning_loop()
elif args.algo == "random":
    Random = RandomSearch(EXPT_NAME, CONFIG, tuning_config, TrainingLoop)
    Random.tuning_loop(RUNS)
else:
    raise ValueError("Choose one of 'grid', 'random'")
