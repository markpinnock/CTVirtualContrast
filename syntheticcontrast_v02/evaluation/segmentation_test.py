import argparse
import numpy as np
import os
import tensorflow as tf
import yaml

from .unet import UNet
from .util import load_data, dice_loss

#-------------------------------------------------------------------------

def test(expt_path):

    with open(f"{expt_path}/config.yml", 'r') as infile:
        config = yaml.load(infile, yaml.FullLoader)

    test_ds = tf.data.Dataset.from_tensor_slices(
        load_data("test", config["model_config"]["img_dims"][0], **config["data_config"]))

    segs = next(iter(test_ds.batch(256)))[1]

    test_ds = test_ds.batch(config["model_config"]["mb_size"])

    Model = UNet(config["model_config"])
    Model.load_weights(f"{expt_path}/models/model.ckpt").expect_partial()
    prediction = Model.predict(test_ds)

    dice = 1 - dice_loss(prediction, segs).numpy()

    return dice


#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Segmentation-based evaluation testing routine """

    expts_path = "syntheticcontrast_v02/evaluation/expts"
    expts = os.listdir(expts_path)

    results = {}

    for expt in expts:
        results[expt] = test(f"{expts_path}/{expt}")

    for k, v in results.items():
        print(k, v)
