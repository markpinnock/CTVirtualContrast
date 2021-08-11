import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.metrics as skim
import tensorflow.keras as keras
import tensorflow as tf

from networks.model import GAN, CropGAN
from networks.UNet import UNet, CropUNet
from utils.dataloader import OneToOneLoader, ManyToOneLoader


""" Inference script """

# Handle arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", "-cp", help="Config json path", type=str)
parser.add_argument("--expt_name", "-en", help="Expt name", type=str)
parser.add_argument("--show_images", "-s", help="Show images y/n", type=str, default='y')
arguments = parser.parse_args()

# Parse config json
with open(arguments.config_path, 'r') as infile:
    CONFIG = json.load(infile)

assert arguments.expt_name, "Provide expt name"
EXPT_NAME = arguments.expt_name
ROI = int(EXPT_NAME.split('_')[0].strip('A'))

if arguments.show_images == 'y':
    show_flag = True
else:
    show_flag = False

CONFIG["EXPT"]["IMG_DIMS"] = [ROI, ROI, 12]
CONFIG["EXPT"]["DATA_PATH"] = CONFIG["EXPT"]["DATA_PATH"].strip('/') + "Test/"
CONFIG["EXPT"]["CV_FOLDS"] = 1

CROP_CONFIG = {k: v for k, v in CONFIG.items()}
CROP_CONFIG["EXPT"]["CROP"] = 1
CROP_CONFIG["EXPT"]["IMG_DIMS"] = [64, 64, 12]

if ROI < 256:
    CONFIG["EXPT"]["CROP"] = 1
else:
    CONFIG["EXPT"]["CROP"] = 0

if 'r' in CONFIG["EXPT"]["DATA"]["INPUTS"]:
    assert 'r' in CONFIG["EXPT"]["DATA"]["LABELS"]
    assert 'r' in CONFIG["EXPT"]["DATA"]["SEGS"]
    assert 'r' in CONFIG["EXPT"]["DATA"]["JSON"]
    Loader = ManyToOneLoader

else:
    Loader = OneToOneLoader

MODEL_PATH = f"{CONFIG['EXPT']['SAVE_PATH']}models/{CONFIG['EXPT']['MODEL']}/{EXPT_NAME}/{EXPT_NAME}"
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

# Compile model
if CONFIG["EXPT"]["MODEL"] == "UNet":
    if not CONFIG["EXPT"]["CROP"]:
        Model = UNet(config=CONFIG)
        Cropper = CropUNet(config=CROP_CONFIG)
    elif CONFIG["EXPT"]["CROP"]:
        Model = CropUNet(config=CONFIG)

elif CONFIG["EXPT"]["MODEL"] == "GAN":
    if not CONFIG["EXPT"]["CROP"]:
        Model = GAN(config=CONFIG)
        Cropper = CropGAN(config=CROP_CONFIG)
    elif CONFIG["EXPT"]["CROP"]:
        Model = CropGAN(config=CONFIG)

Model.load_weights(f"{MODEL_PATH}")

base_MSE = []
base_pSNR = []
base_SSIM = []
test_MSE = []
test_pSNR = []
test_SSIM = []

count = 0

ACE_intensities = [[], [], []]
NCE_intensities = [[], [], []]
pred_intensities = [[], [], []]

for data in test_ds:
    NCE, ACE, seg, coords = data

    if CONFIG["EXPT"]["CROP"]:
        NCEs = []
        ACEs = []

        for i in range(3):
            rNCE, rACE, _ = Model.crop_ROI(NCE, ACE, seg, coords[:, i, :])
            NCEs.append(rNCE[0, ...])
            ACEs.append(rACE[0, ...])
    
        NCE = tf.stack(NCEs, axis=0)
        ACE = tf.stack(ACEs, axis=0)

        pred = Model(NCE, training=False).numpy()
        NCE = NCE.numpy()
        ACE = ACE.numpy()
    
    else:
        NCEs = []
        ACEs = []
        preds = []

        pred = Model(NCE, training=False)

        for i in range(3):
            rNCE, rACE, rpred = Cropper.crop_ROI(NCE, ACE, pred, coords[:, i, :])
            NCEs.append(rNCE[0, ...])
            ACEs.append(rACE[0, ...])
            preds.append(rpred[0, ...])
    
        NCE = tf.stack(NCEs, axis=0).numpy()
        ACE = tf.stack(ACEs, axis=0).numpy()
        pred = tf.stack(preds, axis=0).numpy()

    if CONFIG["EXPT"]["NORM_TYPE"] == "-11":
        pred = (pred + 1) / 2
        NCE = (NCE + 1) / 2
        ACE = (ACE + 1) / 2

    for i in range(ACE.shape[0]):
        base_MSE.append(skim.mean_squared_error(ACE[i, :, :, :, 0], NCE[i, :, :, :, 0]))
        base_pSNR.append(skim.peak_signal_noise_ratio(ACE[i, :, :, :, 0], NCE[i, :, :, :, 0], data_range=ACE[i, :, :, :, 0].max() - ACE[i, :, :, :, 0].min()))
        base_SSIM.append(skim.structural_similarity(ACE[i, :, :, :, 0], NCE[i, :, :, :, 0]))
        test_MSE.append(skim.mean_squared_error(ACE[i, :, :, :, 0], pred[i, :, :, :, 0]))
        test_pSNR.append(skim.peak_signal_noise_ratio(ACE[i, :, :, :, 0], pred[i, :, :, :, 0], data_range=ACE[i, :, :, :, 0].max() - ACE[i, :, :, :, 0].min()))
        test_SSIM.append(skim.structural_similarity(ACE[i, :, :, :, 0], pred[i, :, :, :, 0]))

    if show_flag:
        NCE = NCE * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN
        ACE = ACE * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN
        pred = pred * (ABDO_WINDOW_MAX - ABDO_WINDOW_MIN) + ABDO_WINDOW_MIN

        print(img_list[count])

        fig, axs = plt.subplots(ACE.shape[0], 4)

        for i in range(ACE.shape[0]):
            c = coords[0, i, :] / 4

            if i == 0 and "T028" in img_list[count] and "000.npy" in img_list[count]:
                c = [21, 30]
                offset = 6
            elif i == 0 and "T028" in img_list[count] and "001.npy" in img_list[count]:
                c = [26, 37]
                offset = 6
            elif i == 0 and "T028" in img_list[count] and "002.npy" in img_list[count]:
                c = [28, 34]
                offset = 6
            if i == 0 and "T028" in img_list[count]:
                offset = 6
                axs[i, 0].imshow(NCE[i, c[0] - offset:c[0] + offset, c[1] - offset:c[1] + offset, 11, 0].T, cmap="gray")
                axs[i, 0].axis("off")
                # axs[i, 0].set_title(f"{base_MSE[-ACE.shape[0] + i]:.3f} {base_pSNR[-ACE.shape[0] + i]:.3f} {base_SSIM[-ACE.shape[0] + i]:.3f}")
                axs[i, 1].imshow(ACE[i, c[0] - offset:c[0] + offset, c[1] - offset:c[1] + offset, 11, 0].T, cmap="gray")
                axs[i, 1].axis("off")
                axs[i, 1].set_title(img_list[count])
                axs[i, 2].imshow(pred[i, c[0] - offset:c[0] + offset, c[1] - offset:c[1] + offset, 11, 0].T, cmap="gray")
                axs[i, 2].axis("off")
                # axs[i, 2].set_title(f"{test_MSE[-ACE.shape[0] + i]:.3f} {test_pSNR[-ACE.shape[0] + i]:.3f} {test_SSIM[-ACE.shape[0] + i]:.3f}")
                # axs[i, 3].imshow(np.abs(pred[i, :, :, 11, 0].T - ACE[i, :, :, 11, 0].T), cmap="hot")
                axs[i, 3].imshow(NCE[i, :, :, 11, 0].T, cmap="gray")
                axs[i, 3].axis("off")
            elif i == 0 and "T029" in img_list[count]:
                offset = 8
                axs[i, 0].imshow(NCE[i, c[0] - offset:c[0] + offset, c[1] - offset:c[1] + offset, 11, 0].T, cmap="gray")
                axs[i, 0].axis("off")
                # axs[i, 0].set_title(f"{base_MSE[-ACE.shape[0] + i]:.3f} {base_pSNR[-ACE.shape[0] + i]:.3f} {base_SSIM[-ACE.shape[0] + i]:.3f}")
                axs[i, 1].imshow(ACE[i, c[0] - offset:c[0] + offset, c[1] - offset:c[1] + offset, 11, 0].T, cmap="gray")
                axs[i, 1].axis("off")
                axs[i, 1].set_title(img_list[count])
                axs[i, 2].imshow(pred[i, c[0] - offset:c[0] + offset, c[1] - offset:c[1] + offset, 11, 0].T, cmap="gray")
                axs[i, 2].axis("off")
                # axs[i, 2].set_title(f"{test_MSE[-ACE.shape[0] + i]:.3f} {test_pSNR[-ACE.shape[0] + i]:.3f} {test_SSIM[-ACE.shape[0] + i]:.3f}")
                # axs[i, 3].imshow(np.abs(pred[i, :, :, 11, 0].T - ACE[i, :, :, 11, 0].T), cmap="hot")
                axs[i, 3].imshow(NCE[i, :, :, 11, 0].T, cmap="gray")
                axs[i, 3].axis("off")
            else:
                offset = 16
                axs[i, 0].imshow(NCE[i, offset:-offset, offset:-offset, 11, 0].T, cmap="gray")
                axs[i, 0].axis("off")
                # axs[i, 0].set_title(f"{base_MSE[-ACE.shape[0] + i]:.3f} {base_pSNR[-ACE.shape[0] + i]:.3f} {base_SSIM[-ACE.shape[0] + i]:.3f}")
                axs[i, 1].imshow(ACE[i, offset:-offset, offset:-offset, 11, 0].T, cmap="gray")
                axs[i, 1].axis("off")
                axs[i, 1].set_title(img_list[count])
                axs[i, 2].imshow(pred[i, offset:-offset, offset:-offset, 11, 0].T, cmap="gray")
                axs[i, 2].axis("off")
                # axs[i, 2].set_title(f"{test_MSE[-ACE.shape[0] + i]:.3f} {test_pSNR[-ACE.shape[0] + i]:.3f} {test_SSIM[-ACE.shape[0] + i]:.3f}")
                # axs[i, 3].imshow(np.abs(pred[i, :, :, 11, 0].T - ACE[i, :, :, 11, 0].T), cmap="hot")
                axs[i, 3].imshow(NCE[i, :, :, 11, 0].T, cmap="gray")
                axs[i, 3].axis("off")
       
            if i == 1 and "T028" in img_list[count]:
                print("Skipping T028 right kidney")
                continue
    
            if i == 1 and "T029" in img_list[count] and img_list[count][-7:] in ["000.npy", "001.npy", "002.npy"]:
                print("Skipping T029 right kidney")
                continue

            if i == 2 and "T029" in img_list[count] and img_list[count][-7:] in ["000.npy"]:
                print("Skipping T029 left kidney")
                continue
        
            if img_list[count][:-8] in ["T031A0HQ002", "T031A0HQ003", "T031A0HQ005"]:
                print(f"Skipping {img_list[count][:-8]}")
                continue

            NCE_intensities[i].append(NCE[i, offset:-offset, offset:-offset, 11, 0].ravel())
            ACE_intensities[i].append(ACE[i, offset:-offset, offset:-offset, 11, 0].ravel())
            pred_intensities[i].append(pred[i, offset:-offset, offset:-offset, 11, 0].ravel())

        plt.tight_layout()
        # plt.pause(0.5)
        plt.close()

        count += 1

print("Aorta")
print(np.median(np.hstack(NCE_intensities[0])), np.median(np.hstack(ACE_intensities[0])), np.median(np.hstack(pred_intensities[0])))
# print(np.quantile(np.hstack(NCE_intensities[0]), [0.05, 0.95]), np.quantile(np.hstack(ACE_intensities[0]), [0.05, 0.95]), np.quantile(np.hstack(pred_intensities[0]), [0.05, 0.95]))
print("R Kidney")
print(np.median(np.hstack(NCE_intensities[1])), np.median(np.hstack(ACE_intensities[1])), np.median(np.hstack(pred_intensities[1])))
# print(np.quantile(np.hstack(NCE_intensities[1]), [0.05, 0.95]), np.quantile(np.hstack(ACE_intensities[1]), [0.05, 0.95]), np.quantile(np.hstack(pred_intensities[1]), [0.05, 0.95]))
print("L Kidney")
print(np.median(np.hstack(NCE_intensities[2])), np.median(np.hstack(ACE_intensities[2])), np.median(np.hstack(pred_intensities[2])))
# print(np.quantile(np.hstack(NCE_intensities[2]), [0.05, 0.95]), np.quantile(np.hstack(ACE_intensities[2]), [0.05, 0.95]), np.quantile(np.hstack(pred_intensities[2]), [0.05, 0.95]))

plt.subplot(3, 3, 1)
plt.hist(np.hstack(NCE_intensities[0]).ravel(), bins=50)
plt.title("NCE, aorta")
plt.subplot(3, 3, 2)
plt.hist(np.hstack(ACE_intensities[0]).ravel(), bins=50)
plt.title("ACE, aorta")
plt.subplot(3, 3, 3)
plt.hist(np.hstack(pred_intensities[0]).ravel(), bins=50)
plt.title("Pred, aorta")
plt.subplot(3, 3, 4)
plt.hist(np.hstack(NCE_intensities[1]).ravel(), bins=50)
plt.title("NCE, R kidney")
plt.subplot(3, 3, 5)
plt.hist(np.hstack(ACE_intensities[1]).ravel(), bins=50)
plt.title("ACE, R kidney")
plt.subplot(3, 3, 6)
plt.hist(np.hstack(pred_intensities[1]).ravel(), bins=50)
plt.title("Pred, R kidney")
plt.subplot(3, 3, 7)
plt.hist(np.hstack(NCE_intensities[2]).ravel(), bins=50)
plt.title("NCE, L kidney")
plt.subplot(3, 3, 8)
plt.hist(np.hstack(ACE_intensities[2]).ravel(), bins=50)
plt.title("ACE, L kidney")
plt.subplot(3, 3, 9)
plt.hist(np.hstack(pred_intensities[2]).ravel(), bins=50)
plt.title("Pred, L kidney")
plt.show()

print(f"Baseline: {np.median(base_MSE), np.median(base_pSNR), np.median(base_SSIM)}")
# # print(f"Baseline: {np.quantile(base_MSE, [0.05, 0.95]), np.quantile(base_pSNR, [0.05, 0.95]), np.quantile(base_SSIM, [0.05, 0.95])}")
# print(f"Predicted: {np.median(test_MSE), np.median(test_pSNR), np.median(test_SSIM)}")
# # print(f"Predicted: {np.quantile(test_MSE, [0.05, 0.95]), np.quantile(test_pSNR, [0.05, 0.95]), np.quantile(test_SSIM, [0.05, 0.95])}")
    