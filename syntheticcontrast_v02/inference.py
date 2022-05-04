import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import yaml

from syntheticcontrast_v02.networks.pix2pix import Pix2Pix, HyperPix2Pix
from syntheticcontrast_v02.networks.cyclegan import CycleGAN
from syntheticcontrast_v02.networks.unet import UNet
from syntheticcontrast_v02.utils.build_dataloader import get_dataloader


#-------------------------------------------------------------------------

def inference(CONFIG, phase, save):
    assert phase in ["AC", "VC", "both"], phase

    test_ds, TestGenerator = get_dataloader(config=CONFIG,
                                            dataset="test",
                                            mb_size=64,
                                            stride_length=16)

    # Create model and load weights
    if CONFIG["expt"]["model"] == "Pix2Pix":
        # Hack to avoid triggering assertion in pix2pix.py
        CONFIG["data"]["times"] = []
        Model = Pix2Pix(config=CONFIG)

    elif CONFIG["expt"]["model"] == "HyperPix2Pix":
        Model = HyperPix2Pix(config=CONFIG)

    elif CONFIG["expt"]["model"] == "CycleGAN":
        Model = CycleGAN(config=CONFIG)

    elif CONFIG["expt"]["model"] == "UNet":
        Model = UNet(config=CONFIG)

    else:
        raise ValueError(f"Invalid model type: {CONFIG['expt']['model']}")

    if CONFIG["expt"]["model"] == "CycleGAN":
        Model.G_forward.load_weights(f"{CONFIG['paths']['expt_path']}/models/generator.ckpt")
    elif CONFIG["expt"]["model"] == "UNet":
        Model.UNet.load_weights(f"{CONFIG['paths']['expt_path']}/models/model.ckpt")
    else:
        Model.Generator.load_weights(f"{CONFIG['paths']['expt_path']}/models/generator.ckpt")

    AC_predictions = {}
    VC_predictions = {}
    weights = {}

    for data in test_ds:
        if CONFIG["expt"]["model"] == "CycleGAN":
            AC_pred = Model.G_forward(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * 1.0)
            VC_pred = Model.G_forward(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * 2.0)
        elif CONFIG["expt"]["model"] == "UNet":
            AC_pred = Model.UNet(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * 1.0)
            VC_pred = Model.UNet(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * 2.0)
        else:
            AC_pred = Model.Generator(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * 1.0)
            VC_pred = Model.Generator(data["real_source"], tf.ones([data["real_source"].shape[0], 1]) * 2.0)

        AC_pred = TestGenerator.un_normalise(AC_pred)[:, :, :, :, 0].numpy()
        VC_pred = TestGenerator.un_normalise(VC_pred)[:, :, :, :, 0].numpy()
        AC_pred = np.round(AC_pred).astype("int16")
        VC_pred = np.round(VC_pred).astype("int16")

        for i in range(AC_pred.shape[0]):
            subject_ID = data["subject_ID"].numpy()[i].decode("utf-8")[:-4]
            x_slice = slice(int(data["x"][i]), int(data["x"][i]) + int(CONFIG["data"]["patch_size"][0]))
            y_slice = slice(int(data["y"][i]), int(data["y"][i]) + int(CONFIG["data"]["patch_size"][1]))

            if int(data["z"][i]) < 0:
                z_slice = slice(int(data["z"][i]), None)
            else:
                z_slice = slice(int(data["z"][i]), int(data["z"][i]) + int(CONFIG["data"]["patch_size"][2]))

            if subject_ID not in AC_predictions.keys():
                print(subject_ID)
                original_img = glob.glob(f"{CONFIG['data']['data_path']}/Images/{subject_ID}*")[0]
                img_dims = np.load(original_img).shape
                weights[subject_ID] = np.zeros(img_dims)
                AC_predictions[subject_ID] = np.zeros(img_dims)
                VC_predictions[subject_ID] = np.zeros(img_dims)

            AC_predictions[subject_ID][x_slice, y_slice, z_slice] += AC_pred[i, :, :, :]
            VC_predictions[subject_ID][x_slice, y_slice, z_slice] += VC_pred[i, :, :, :]
            weights[subject_ID][x_slice, y_slice, z_slice] += 1

    for subject_ID, w in weights.items():
        AC = AC_predictions[subject_ID] / w
        VC = VC_predictions[subject_ID] / w

        if save:
            save_path = f"{CONFIG['paths']['expt_path']}/predictions"
            print(f"{subject_ID} saved")
            if not os.path.exists(save_path): os.mkdir(save_path)

            if phase == "AC":
                np.save(f"{save_path}/{subject_ID[0:6]}AP{subject_ID[-3:]}", AC)
            elif phase == "VC":
                np.save(f"{save_path}/{subject_ID[0:6]}VP{subject_ID[-3:]}", VC)
            elif phase == "both":
                np.save(f"{save_path}/{subject_ID[0:6]}AP{subject_ID[-3:]}", AC)
                np.save(f"{save_path}/{subject_ID[0:6]}VP{subject_ID[-3:]}", VC)
            else:
                raise ValueError

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
    parser.add_argument("--data", '-d', help="Data path", type=str)
    parser.add_argument("--phase", '-f', help="Phase: AC/VC/both", type=str)
    parser.add_argument("--save", '-s', help="Save images", action="store_true")
    arguments = parser.parse_args()

    EXPT_PATH = arguments.path

    # Parse config json
    with open(f"{EXPT_PATH}/config.yml", 'r') as infile:
        CONFIG = yaml.load(infile, yaml.FullLoader)
    
    CONFIG["paths"]["expt_path"] = arguments.path
    CONFIG["data"]["data_path"] = arguments.data
    
    inference(CONFIG, arguments.phase, arguments.save)
