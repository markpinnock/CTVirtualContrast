import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf


def imgLoader(file_path, mb_size, shuffle):
    CE_path = file_path.decode("utf-8")  + 'ACE/'
    NCE_path = file_path.decode("utf-8")  + 'NCE/'
    CE_list = os.listdir(CE_path)
    NCE_list = os.listdir(NCE_path)
    CE_list.sort()
    NCE_list.sort()

    if shuffle == True:
        temp_list = list(zip(CE_list, NCE_list))
        np.random.shuffle(temp_list)
        CE_list, NCE_list = zip(*temp_list)

    N = len(CE_list)
    i = 0

    while i < int(N / mb_size):
        CE_mb = CE_list[i*mb_size:(i+1)*mb_size]
        NCE_mb = NCE_list[i*mb_size:(i+1)*mb_size]
        CE_img = [np.load(CE_path + img) for img in CE_mb]
        NCE_img = [np.load(NCE_path + img) for img in NCE_mb]
        CE_img = [img[::4, ::4, ::2] for img in CE_img]
        NCE_img = [img[::4, ::4, ::2] for img in NCE_img]

        i += 1
        yield CE_img, NCE_img


if __name__ == "__main__":
    FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/tf2_prac/train/"
    data = tf.data.Dataset.from_generator(imgLoader, args=[FILE_PATH, 4, True], output_types=tf.float32)

    for imgs in data:
        CE = imgs[0, :, :, :, :, tf.newaxis]
        NCE = imgs[1, :, :, :, :, tf.newaxis]
        print(CE.shape, NCE.shape)
