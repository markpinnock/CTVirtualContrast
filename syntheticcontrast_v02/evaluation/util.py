from matplotlib.pyplot import imshow
import nrrd
import numpy as np
import os
import tensorflow as tf


#-------------------------------------------------------------------------
""" Dataloader for segmentation-based evaluation - loads images .nrrd
    image volumes into RAM as 2D images """

def load_data(split, img_res, img_path, seg_path, img_type, ignore):
    HU_min = -500
    HU_max = 2500
    unique_ids = []
    train_test_split = 10

    for img_id in os.listdir(img_path):
        if img_id[0:4] not in unique_ids and img_id[0:6] != ignore:
            unique_ids.append(img_id[0:4])

    np.random.seed(5)
    np.random.shuffle(unique_ids)

    if split == "train":
        fold_ids = unique_ids[0:train_test_split]
    elif split == "test":
        fold_ids = unique_ids[train_test_split:]
    else:
        raise ValueError

    img_list = [img for img in os.listdir(img_path) if img_type in img and img[0:4] in fold_ids]
    seg_list = []

    for img in img_list:
        if img_type in ["AC", "VC"]:
            seg_list.append(f"{img.split('.')[0]}-label.nrrd")
        else:
            seg_list.append(f"{img[0:6]}HQ{img[8:11]}-label.nrrd")

    imgs = []
    segs = []

    for img, seg in zip(img_list, seg_list):
        img_arr = nrrd.read(f"{img_path}/{img}")[0].astype("float32")
        seg_arr = nrrd.read(f"{seg_path}/{seg}")[0].astype("float32")
        img_dims = img_arr.shape
        img_arr[img_arr < HU_min] = HU_min
        img_arr[img_arr > HU_max] = HU_max
        img_arr = (img_arr - HU_min) / (HU_max - HU_min)

        idx = np.argwhere(seg_arr == 1)
        x = (np.unique(idx[:, 0])[0], np.unique(idx[:, 0])[-1])
        y = (np.unique(idx[:, 1])[0], np.unique(idx[:, 1])[-1])
        z = (np.unique(idx[:, 2])[0], np.unique(idx[:, 2])[-1])
        padding_x = img_res - (x[1] - x[0] + 1)
        padding_y = img_res - (y[1] - y[0] + 1)
        x_padded = [x[0] - padding_x // 2, x[1] + padding_x // 2]
        y_padded = [y[0] - padding_y // 2, y[1] + padding_y // 2]

        if padding_x % 2 != 0:
            x_padded[1] += 1

        if padding_y % 2 != 0:
            y_padded[1] += 1

        if x_padded[1] > img_dims[0] - 1:
            x_padded[0] -= (x_padded[1] - img_dims[0] + 1)
            x_padded[1] -= (x_padded[1] - img_dims[0] + 1)
        elif x_padded[0] < 0:
            x_padded[1] -= x_padded[0]
            x_padded[0] -= x_padded[0]
        else:
            pass

        if y_padded[1] > img_dims[1] - 1:
            y_padded[0] -= (y_padded[1] - img_dims[1] + 1)
            y_padded[1] -= (y_padded[1] - img_dims[1] + 1)
        elif y_padded[0] < 0:
            y_padded[1] += y_padded[0]
            y_padded[0] += y_padded[0]
        else:
            pass

        for i in range(*z):
            imgs.append(img_arr[x_padded[0]:(x_padded[1] + 1), y_padded[0]:(y_padded[1] + 1), i][:, :, np.newaxis])
            segs.append(seg_arr[x_padded[0]:(x_padded[1] + 1), y_padded[0]:(y_padded[1] + 1), i][:, :, np.newaxis])

    return imgs, segs


#-------------------------------------------------------------------------
""" Data augmentation layer """

class Augmentation(tf.keras.layers.Layer):
    def __init__(self, img_res, seed=5):
        super().__init__(name="augmentation")

        self.image_augment = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed),
            tf.keras.layers.RandomZoom(0.2, seed=seed),
            tf.keras.layers.RandomCrop(img_res[0], img_res[1], seed=seed)
        ]

        self.label_augment = [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.2, seed=seed, interpolation="nearest"),
            tf.keras.layers.RandomZoom(0.2, seed=seed, interpolation="nearest"),
            tf.keras.layers.RandomCrop(img_res[0], img_res[1], seed=seed)
        ]

    def call(self, x, y):
        for aug in self.image_augment:
            x = aug(x)
        
        for aug in self.label_augment:
            y = aug(y)

        return x, y


#-------------------------------------------------------------------------
""" Sorenson-Dice loss for segmentation-based evaluation """

def dice_loss(A, B):
    numerator = 2 * tf.reduce_sum(A * B)
    denominator = tf.reduce_sum(A) + tf.reduce_sum(B) + 1e-12

    return 1 - numerator / denominator


#-------------------------------------------------------------------------
""" Routine for testing data loader, augmentation and Dice """

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    imgs, segs = load_data(
        "test",
        "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real/Images",
        "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real/Segmentations_Tumour",
        "VC",
        "T065A1",
        100
    )

    ds = tf.data.Dataset.from_tensor_slices((imgs, segs)).shuffle(256).batch(4).map(Augmentation(64))

    for img, seg in ds:
        fig, axs = plt.subplots(2, 4)

        for i in range(4):
            print(dice_loss(seg[i, ...], seg[i, ...]))
            axs[0, i].imshow(img[i, ...], cmap="gray")
            axs[1, i].imshow(seg[i, ...], cmap="gray")

        plt.show()
