import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import yaml

from syntheticcontrast_v02.networks.pix2pix import Pix2Pix, HyperPix2Pix
from syntheticcontrast_v02.networks.cyclegan import CycleGAN
from syntheticcontrast_v02.utils.dataloader_v02 import PairedLoader, UnpairedLoader


#-------------------------------------------------------------------------

def inference(CONFIG, save):

    if CONFIG["data"]["data_type"] == "paired":
        Loader = PairedLoader

    elif CONFIG["data"]["data_type"] == "unpaired":
        Loader = UnpairedLoader

    else:
        raise ValueError("Select paired or unpaired dataloader")

    # Initialise datasets and set normalisation parameters
    CONFIG["data"]["cv_folds"] = 1
    CONFIG["data"]["fold"] = 0
    CONFIG["data"]["segs"] = []
    CONFIG["data"]["data_path"] += "Test"
    CONFIG["data"]["stride_length"] = 32
    TestGenerator = Loader(config=CONFIG["data"], dataset_type="validation")
    _, _ = TestGenerator.set_normalisation()

    # Specify output types
    output_types = {"real_source": "float32", "subject_ID": tf.string, "x": "int32", "y": "int32", "z": "int32"}

    # Create dataloader
    test_ds = tf.data.Dataset.from_generator(
        generator=TestGenerator.inference_generator,
        output_types=output_types).batch(1)

    # Create model and load weights
    if CONFIG["expt"]["model"] == "Pix2Pix":
        Model = Pix2Pix(config=CONFIG)

    elif CONFIG["expt"]["model"] == "HyperPix2Pix":
        Model = HyperPix2Pix(config=CONFIG)

    elif CONFIG["expt"]["model"] == "CycleGAN":
        Model = CycleGAN(config=CONFIG)

    else:
        raise ValueError(f"Invalid model type: {CONFIG['expt']['model']}")
    
    Model.Generator.load_weights(f"{CONFIG['paths']['expt_path']}/models/generator.ckpt")

    AC_predictions = {}
    VC_predictions = {}
    weights = {}
    # vol_depths = {}

    for data in test_ds:
        AC_pred = Model.Generator(data["real_source"], tf.ones([1, 1]) * 1.0)
        VC_pred = Model.Generator(data["real_source"], tf.ones([1, 1]) * 2.0)
        AC_pred = TestGenerator.un_normalise(AC_pred)[0, :, :, :, 0].numpy()
        VC_pred = TestGenerator.un_normalise(VC_pred)[0, :, :, :, 0].numpy()
        AC_pred = np.round(AC_pred).astype("int16")
        VC_pred = np.round(VC_pred).astype("int16")

        subject_ID = data["subject_ID"].numpy()[0].decode("utf-8")[:-4]
        x_slice = slice(data["x"], data["x"] + CONFIG["data"]["patch_size"][0])
        y_slice = slice(data["y"], data["y"] + CONFIG["data"]["patch_size"][1])
        z_slice = slice(data["z"], data["z"] + CONFIG["data"]["patch_size"][2])

        if subject_ID not in AC_predictions.keys():
            original_img = glob.glob(f"{CONFIG['data']['data_path']}/Images/{subject_ID}*")[0]
            img_dims = np.load(original_img).shape
            weights[subject_ID] = np.zeros(img_dims)
            AC_predictions[subject_ID] = np.zeros(img_dims)
            VC_predictions[subject_ID] = np.zeros(img_dims)
            # vol_depths[subject_ID] = []

        AC_predictions[subject_ID][x_slice, y_slice, z_slice]
        VC_predictions[subject_ID][x_slice, y_slice, z_slice]
        weights[subject_ID][x_slice, y_slice, z_slice] += 1
        # vol_depths[subject_ID].append([data["index_low"].numpy().astype("int16")[0], data["index_high"].numpy().astype("int16")[0]])

    for subject, w in weights.items():
        # AC = np.zeros((256, 256, indices[-1][-1]), dtype="int16")
        # VC = np.zeros((256, 256, indices[-1][-1]), dtype="int16")

        # for i, index in enumerate(indices):
        #     AC[:, :, index[0]:index[1]] = AC_predictions[subject][i]
        #     VC[:, :, index[0]:index[1]] = VC_predictions[subject][i]

        AC = AC_predictions[subject_ID] / w
        VC = VC_predictions[subject_ID] / w

        if save:
            save_path = f"{CONFIG['paths']['expt_path']}/predictions"
            if not os.path.exists(save_path): os.mkdir(save_path)
            np.save(f"{save_path}/{subject[0:6]}AP{subject[-3:]}", AC)
            np.save(f"{save_path}/{subject[0:6]}VP{subject[-3:]}", VC)

        else:
            plt.subplot(2, 2, 1)
            plt.imshow(AC[:, :, 32], cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(np.flipud(AC[128, :, :].T), cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(VC[:, :, 32], cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(np.flipud(VC[128, :, :].T), cmap="gray", vmin=-150, vmax=250)
            plt.axis("off")
            plt.show()
    

#-------------------------------------------------------------------------

if __name__ == "__main__":

    """ Inference routine """

    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Expt path", type=str)
    parser.add_argument("--save", "-s", help="Visualise", action="store_true")
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)
    
    CONFIG["paths"]["expt_path"] = arguments.path
    
    inference(CONFIG, arguments.save)
