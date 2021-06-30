import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras as keras
import tensorflow as tf

from networks.GANWrapper import GAN, CropGAN
from networks.UNet import UNet, CropUNet
from utils.DataLoader import OneToOneLoader, ManyToOneLoader

""" Img generation """

# Parse config json
with open("UNet.json", 'r') as infile:
    CONFIG = json.load(infile)

expts = [
    "A256_000_0000e-0",
    "A256_055_0000e-0",
    "A256_000_1946e-5",
    "A256_055_1946e-5",
    "A64_000_0000e-0",
    "A64_045_0000e-0",
    "A64_000_5200e-6",
    "A64_045_5200e-6"
    ]

ablations = [
    "CE",
    "NCE",
    "UNet",
    "UNetF",
    "UNetW",
    "UNetFW",
    "UNetC",
    "UNetCF",
    "UNetCW",
    "UNetCFW"
]

CONFIG["EXPT"]["IMG_DIMS"] = [256, 256, 12]
CONFIG["EXPT"]["CROP"] = 0
CONFIG["EXPT"]["DATA_PATH"] = CONFIG["EXPT"]["DATA_PATH"].strip('/') + "Subset/"
CONFIG["EXPT"]["CV_FOLDS"] = 1

CROP_CONFIG = {k: v for k, v in CONFIG.items()}
CROP_CONFIG["EXPT"]["CROP"] = 1
CROP_CONFIG["EXPT"]["IMG_DIMS"] = [64, 64, 12]

if 'r' in CONFIG["EXPT"]["DATA"]["INPUTS"]:
    assert 'r' in CONFIG["EXPT"]["DATA"]["LABELS"]
    assert 'r' in CONFIG["EXPT"]["DATA"]["SEGS"]
    assert 'r' in CONFIG["EXPT"]["DATA"]["JSON"]
    Loader = ManyToOneLoader

else:
    Loader = OneToOneLoader

img_list = os.listdir(f"{CONFIG['EXPT']['DATA_PATH']}rHQ")

TestGenerator = Loader(
    config=CONFIG["EXPT"],
    dataset_type="validation",
    fold=0)

test_ds = tf.data.Dataset.from_generator(
    generator=TestGenerator.data_generator,
    output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
    ).batch(1)

ABDO_WINDOW_MIN = -150
ABDO_WINDOW_MAX = 250

# Compile models
models = dict.fromkeys(expts)
outputs = dict.fromkeys(["CE", "NCE"] + expts)

for expt in models.keys():
    model_path = f"{CONFIG['EXPT']['SAVE_PATH']}models/{CONFIG['EXPT']['MODEL']}/{expt}/{expt}"

    if CONFIG["EXPT"]["MODEL"] == "UNet":
        if "256" in expt:
            models[expt] = UNet(config=CONFIG)
        else:
            models[expt] = CropUNet(config=CROP_CONFIG)

    elif CONFIG["EXPT"]["MODEL"] == "GAN":
        if "256" in expt:
            models[expt] = GAN(config=CONFIG)
        else:
            models[expt] = CropGAN(config=CROP_CONFIG)

    models[expt].load_weights(f"{model_path}")

count = 0

for data in test_ds:
    NCE, ACE, seg, coords = data

    for expt, model in models.items():
        if "64" in expt: continue
        preds = []
        pred = model(NCE, training=False)

        for i in range(3):
            _, _, rpred = models["A64_000_0000e-0"].crop_ROI(NCE, ACE, pred, coords[:, i, :])
            preds.append(rpred[0, ...])

        pred = tf.stack(preds, axis=0).numpy()

        if CONFIG["EXPT"]["NORM_TYPE"] == "-11": pred = (pred + 1) / 2
        pred = pred * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN
        outputs[expt] = pred
    
    NCEs = []
    ACEs = []

    for i in range(3):
        rNCE, rACE, _ = models["A64_000_0000e-0"].crop_ROI(NCE, ACE, seg, coords[:, i, :])
        NCEs.append(rNCE[0, ...])
        ACEs.append(rACE[0, ...])
    
    NCE = tf.stack(NCEs, axis=0)
    ACE = tf.stack(ACEs, axis=0)
    outputs["NCE"] = NCE.numpy()
    outputs["CE"] = ACE.numpy()
    
    if CONFIG["EXPT"]["NORM_TYPE"] == "-11":
        outputs["NCE"] = (outputs["NCE"] + 1) / 2
        outputs["CE"] = (outputs["CE"] + 1) / 2

    outputs["CE"] = outputs["CE"] * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN
    outputs["NCE"] = outputs["NCE"] * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN

    for expt, model in models.items():
        if "256" in expt: continue
        pred = model(NCE, training=False).numpy()

        if CONFIG["EXPT"]["NORM_TYPE"] == "-11": pred = (pred + 1) / 2
        pred = pred * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN
        outputs[expt] = pred
    
    fig, axs = plt.subplots(ACE.shape[0], 10, figsize=(18, 5.8))

    for i, img in enumerate(outputs.values()):
        axs[0, i].imshow(img[0, :, :, 11, 0].T, cmap="gray")
        axs[0, i].axis("off")
        axs[0, i].set_title(ablations[i], fontsize=20)
        axs[1, i].imshow(img[1, :, :, 11, 0].T, cmap="gray")
        axs[1, i].axis("off")
        # _, a, _ = axs[1, i].hist(img[0, 22:34, 27:39, 11, 0].T.ravel(), bins=20)
        # print(np.median(a), np.quantile(a, [0.05, 0.95]))
        # axs[1, i].set_xlim([ABDO_WINDOW_MIN, ABDO_WINDOW_MAX])
        # axs[1, i].set_ylim([0, 100])
        axs[2, i].imshow(img[2, :, :, 11, 0].T, cmap="gray")
        axs[2, i].axis("off")

    print(img_list[count])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.show()
    # plt.savefig(f"C:/Users/roybo/OneDrive - University College London/PhD/Publications/003_VirtualContrast/outputs/{img_list[count][:-4]}.png", dpi=500)
    # plt.close()

    count += 1