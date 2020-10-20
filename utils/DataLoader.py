import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf


def img_loader(file_path, shuffle):
    ACE_path = file_path.decode("utf-8")  + "AC/"
    NCE_path = file_path.decode("utf-8")  + "VC/"
    ACE_list = os.listdir(ACE_path)
    NCE_list = os.listdir(NCE_path)
    ACE_list.sort()
    NCE_list.sort()

    if shuffle == True:
        temp_list = list(zip(ACE_list, NCE_list))
        np.random.shuffle(temp_list)
        ACE_list, NCE_list = zip(*temp_list)

    N = len(ACE_list)
    i = 0

    while i < N:
        try:
            ACE_name = ACE_list[i]
            name_start = ACE_name[0:6]
            name_end = ACE_name[-6:]
            ACE_vol = np.load(ACE_path + ACE_name).astype(np.float32)
            NCE_name = glob.glob(f"{NCE_path}{name_start}VC*_{name_end}")
            assert len(NCE_name) == 1
            NCE_vol = np.load(NCE_name[0])

        except Exception as e:
            print(f"IMAGE LOAD FAILURE: {ACE_name} {NCE_name} ({e})")
        
        else:
            ACE_vol = ACE_vol[::4, ::4, :, np.newaxis]
            NCE_vol = NCE_vol[::4, ::4, :, np.newaxis]
            yield (NCE_vol)
        
        finally:
            i += 1


if __name__ == "__main__":
    FILE_PATH = "C:/ProjectImages/VirtualContrast/"
    data = tf.data.Dataset.from_generator(imgLoader, args=[FILE_PATH, True], output_types=(tf.float32, tf.float32)).batch(4)

    for ACE, NCE in data:
        print(ACE.shape, NCE.shape)
