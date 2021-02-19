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
parser.add_argument("--algo", '-a', help="Tuning algorithm", type=str)
parser.add_argument("--expt_name", '-e', help="Experiment name", type=str)
parser.add_argument("--runs", '-r', help="Number of validation runs", type=int)
args = parser.parse_args()

with open(args.config_path, 'r') as infile:
    CONFIG = json.load(infile)

# Set number of runs and run algorithm
if args.runs is not None:
    RUNS = args.runs
else:
    RUNS = 100

if args.expt_name is not None:
    EXPT_NAME = args.expt_name
else:
    EXPT_NAME = "test"

if args.algo == "grid":
    Grid = GridSearch(EXPT_NAME, CONFIG, TrainingLoopUNet)
    Grid.tuning_loop()
elif args.algo == "random":
    Random = RandomSearch(EXPT_NAME, CONFIG, TrainingLoopGAN)
    Random.tuning_loop(RUNS)
else:
    raise ValueError("Choose one of 'grid', 'random'")
