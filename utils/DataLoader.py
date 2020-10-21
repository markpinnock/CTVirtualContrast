import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf


class ImgLoader:
    def __init__(self, file_path, dataset_type, num_folds, fold):
        self.ACE_path = f"{file_path}AC/"
        self.NCE_path = f"{file_path}VC/"
        self.seg_path = f"{file_path}Segs/"
        self.ACE_list = None
        self.NCE_list = None
        self.seg_list = None
        self.dataset_type = dataset_type

        ACE_list = os.listdir(self.ACE_path)
        NCE_list = os.listdir(self.NCE_path)
        seg_list = os.listdir(self.seg_path)

        # TODO: method to return example images

        if num_folds > 0:
            ACE_list.sort()
            NCE_list.sort()
            seg_list.sort()
            np.random.seed(5)
            temp_list = list(zip(ACE_list, NCE_list, seg_list))
            np.random.shuffle(temp_list)
            ACE_list, NCE_list, seg_list = zip(*temp_list)
            num_in_fold = N // num_folds

            if self.dataset_type == "training":
                self.ACE_list = ACE_list[0:fold * num_in_fold] + ACE_list[(fold + 1) * num_in_fold:]
                self.NCE_list = NCE_list[0:fold * num_in_fold] + NCE_list[(fold + 1) * num_in_fold:]
                self.seg_list = seg_list[0:fold * num_in_fold] + seg_list[(fold + 1) * num_in_fold:]
            elif self.dataset_type == "validation":
                self.ACE_list = ACE_list[fold * num_in_fold:(fold + 1) * num_in_fold]
                self.NCE_list = NCE_list[fold * num_in_fold:(fold + 1) * num_in_fold]
                self.seg_list = seg_list[fold * num_in_fold:(fold + 1) * num_in_fold]
            else:
                raise ValueError("Select 'training' or 'validation'")

            np.random.seed()
        
        elif num_folds == 0:
            self.ACE_list = ACE_list
            self.NCE_list = NCE_list
            self.seg_list = seg_list
        
        else:
            raise ValueError("Number of folds must be >= 0")
        
        assert len(ACE_list) == len(NCE_list) and len(ACE_list) == len(seg_list), f"{N} {len(NCE_list)} {len(seg_list)}"
        
    def data_generator(self):
        if self.dataset_type == "training":
            temp_list = list(zip(self.ACE_list, self.NCE_list, self.seg_list))
            np.random.shuffle(temp_list)
            self.ACE_list, self.NCE_list, self.seg_list = zip(*temp_list)

        N = len(self.ACE_list)
        i = 0

        while i < N:
            try:
                ACE_name = self.ACE_list[i]
                NCE_name_start = ACE_name[0:6]
                NCE_name_end = ACE_name[-6:]
                ACE_vol = np.load(self.ACE_path + ACE_name).astype(np.float32)
                NCE_name = glob.glob(f"{self.NCE_path}{NCE_name_start}VC*_{NCE_name_end}")
                assert len(NCE_name) == 1
                seg_name = f"{ACE_name[:-7]}M{ACE_name[-7:]}"
                NCE_vol = np.load(NCE_name[0])
                seg_vol = np.load(f"{self.seg_path}{seg_name}")

            except Exception as e:
                print(f"IMAGE LOAD FAILURE: {ACE_name} {NCE_name} {seg_name} ({e})")
            
            else:
                ACE_vol = ACE_vol[::4, ::4, :, np.newaxis]
                NCE_vol = NCE_vol[::4, ::4, :, np.newaxis]
                seg_vol = seg_vol[::4, ::4, :, np.newaxis]
                yield (NCE_vol, ACE_vol, seg_vol)
            
            finally:
                i += 1


if __name__ == "__main__":
    FILE_PATH = "C:/ProjectImages/VirtualContrast/"
    data = tf.data.Dataset.from_generator(imgLoader, args=[FILE_PATH, True], output_types=(tf.float32, tf.float32)).batch(4)

    for ACE, NCE in data:
        print(ACE.shape, NCE.shape)
