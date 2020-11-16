import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf


class ImgLoader:
    def __init__(self, config, dataset_type, fold):
        file_path = config["DATA_PATH"]
        self.ACE_path = f"{file_path}AC/"
        self.NCE_path = f"{file_path}VC/"
        self.seg_path = f"{file_path}Segs/"
        self.ACE_list = None
        self.NCE_list = None
        self.seg_list = None
        self.dataset_type = dataset_type
        self.down_sample = config["EXPT"]["DOWN_SAMP"]
        self.expt_type = config["EXPT"]["MODEL"]

        ACE_list = os.listdir(self.ACE_path)
        NCE_list = os.listdir(self.NCE_path)
        seg_list = os.listdir(self.seg_path)

        unique_ids = []

        for img_id in ACE_list:
            if img_id[0:4] not in unique_ids:
                unique_ids.append(img_id[0:4])

        N = len(unique_ids)
        # TODO: method to return example images

        if config["EXPT"]["CV_FOLDS"] > 0:
            np.random.seed(5)

            np.random.shuffle(unique_ids)
            num_in_fold = N // config["EXPT"]["CV_FOLDS"]

            if self.dataset_type == "training":
                fold_ids = unique_ids[0:fold * num_in_fold] + unique_ids[(fold + 1) * num_in_fold:]
            elif self.dataset_type == "validation":
                fold_ids = unique_ids[fold * num_in_fold:(fold + 1) * num_in_fold]
            else:
                raise ValueError("Select 'training' or 'validation'")
            
            self.ACE_list = [img_id for img_id in ACE_list if img_id[0:4] in fold_ids]
            self.NCE_list = [img_id for img_id in NCE_list if img_id[0:4] in fold_ids]
            self.seg_list = [img_id for img_id in seg_list if img_id[0:4] in fold_ids]
            
            np.random.seed()
        
        elif config["EXPT"]["CV_FOLDS"] == 0:
            self.ACE_list = ACE_list
            self.NCE_list = NCE_list
            self.seg_list = seg_list
        
        else:
            raise ValueError("Number of folds must be >= 0")

        assert len(self.ACE_list) == len(self.NCE_list) and len(self.ACE_list) == len(self.seg_list), f"{self.ACE_list} {len(self.NCE_list)} {len(self.seg_list)}"
        
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
                ACE_vol = ACE_vol[::self.down_sample, ::self.down_sample, :, np.newaxis]
                NCE_vol = NCE_vol[::self.down_sample, ::self.down_sample, :, np.newaxis]
                seg_vol = seg_vol[::self.down_sample, ::self.down_sample, :, np.newaxis]
                
                if self.expt_type == "GAN":
                    ACE_vol = ACE_vol * 2 - 1
                    NCE_vol = NCE_vol * 2 - 1

                yield (NCE_vol, ACE_vol, seg_vol)
            
            finally:
                i += 1


if __name__ == "__main__":
    FILE_PATH = "C:/ProjectImages/VirtualContrast/"
    data = tf.data.Dataset.from_generator(imgLoader, args=[FILE_PATH, True], output_types=(tf.float32, tf.float32)).batch(4)

    for ACE, NCE in data:
        print(ACE.shape, NCE.shape)
