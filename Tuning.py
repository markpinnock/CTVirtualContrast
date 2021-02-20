import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from TrainingLoops import TrainingLoopUNet, TrainingLoopGAN
from tuners.TuningClasses import GridSearch, RandomSearch

np.set_printoptions(suppress=True)


""" Hyper-parameter tuning script """

# Load configuration json
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", '-cp', help="Path to config json", type=str)
parser.add_argument("--tuning_path", '-tp', help="Path to tuning config json", type=str)
parser.add_argument("--algo", '-a', help="Tuning algorithm", type=str)
parser.add_argument("--expt_name", '-e', help="Experiment name", type=str)
parser.add_argument("--runs", '-r', help="Number of validation runs", type=int)
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
