import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" ImgLoader class: data_generator method for use with tf.data.Dataset.from_generator """

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
        self.down_sample = config["DOWN_SAMP"]
        self.expt_type = config["MODEL"]

        if config["CROP"]:
            self.coords = json.load(open(f"{file_path}coords.json", 'r'))
        else:
            self.coords = None

        ACE_list = os.listdir(self.ACE_path)
        NCE_list = os.listdir(self.NCE_path)
        seg_list = os.listdir(self.seg_path)

        unique_ids = []

        for img_id in ACE_list:
            if img_id[0:4] not in unique_ids:
                unique_ids.append(img_id[0:4])

        N = len(unique_ids)
        # TODO: method to return example images

        if config["CV_FOLDS"] > 0:
            np.random.seed(5)

            np.random.shuffle(unique_ids)
            num_in_fold = N // config["CV_FOLDS"]

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
        
        elif config["CV_FOLDS"] == 0:
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

                if self.coords:
                    yield (NCE_vol, ACE_vol, seg_vol, np.array(self.coords[ACE_name[:-4]]) // self.down_sample)
                else:
                    yield (NCE_vol, ACE_vol, seg_vol)

            finally:
                i += 1


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" DiffAug class for differentiable augmentation
    Paper: https://arxiv.org/abs/2006.10738
    Adapted from: https://github.com/mit-han-lab/data-efficient-gans """

class DiffAug:
    
    def __init__(self, aug_config):
        self.aug_config = aug_config

    """ Random brightness in range [-0.5, 0.5] """
    def brightness(self, x):
        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) - 0.5
        return x + factor
    
    """ Random saturation in range [0, 2] """
    def saturation(self, x):
        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) * 2
        x_mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        return (x - x_mean) * factor + x_mean

    """ Random contrast in range [0.5, 1.5] """
    def contrast(self, x):
        factor = tf.random.uniform((x.shape[0], 1, 1, 1, 1)) + 0.5
        x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        return (x - x_mean) * factor + x_mean
    
    """ Random translation by ratio 0.125 """
    # NB: This assumes NHWDC format and does not (yet) act in z direction
    def translation(self, imgs, seg, ratio=0.125):
        num_imgs = len(imgs)
        batch_size = tf.shape(seg)[0]
        image_size = tf.shape(seg)[1:3]
        image_depth = tf.shape(seg)[3]
        x = tf.concat(imgs + [seg], axis=3)

        shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
        translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
        translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
        grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
        grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
        x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
        x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3, 4]), [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3, 4])
        
        imgs = [x[:, :, :, i * 12:(i + 1) * 12, :] for i in range(num_imgs)]
        seg = x[:, :, :, -12:, :]

        return imgs, seg

    """ Random cutout by ratio 0.5 """
    # NB: This assumes NHWDC format and does not (yet) act in z direction
    def cutout(self, imgs, seg, ratio=0.5):
        num_imgs = len(imgs)
        batch_size = tf.shape(seg)[0]
        image_size = tf.shape(seg)[1:3]
        image_depth = tf.shape(seg)[3]
        x = tf.concat(imgs + [seg], axis=3)

        cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
        offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
        offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)

        grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
        cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)

        mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
        cutout_grid = tf.maximum(cutout_grid, 0)

        cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))

        mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
        x = x * tf.expand_dims(tf.expand_dims(mask, axis=3), axis=4)
       
        imgs = [x[:, :, :, i * 12:(i + 1) * 12, :] for i in range(num_imgs)]
        seg = x[:, :, :, -12:, :]

        return imgs, seg
    
    def augment(self, imgs, seg):
        if self.aug_config["colour"]: imgs = [self.contrast(self.saturation(self.brightness(img))) for img in imgs]
        if self.aug_config["translation"]: imgs, seg = self.translation(imgs, seg)
        if self.aug_config["cutout"]: imgs, seg = self.cutout(imgs, seg)

        return imgs, seg

#----------------------------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == "__main__":

    FILE_PATH = "C:/ProjectImages/VirtualContrast/"
    TestLoader = ImgLoader({"DATA_PATH": FILE_PATH, "DOWN_SAMP": 4, "CV_FOLDS": 6, "MODEL": "GAN", "CROP": 1}, dataset_type="training", fold=0)
    TestAug = DiffAug({"colour": True, "translation": True, "cutout": True})

    train_ds = tf.data.Dataset.from_generator(
        TestLoader.data_generator, output_types=(tf.float32, tf.float32, tf.float32, tf.float32))

    for data in train_ds.batch(4):

        NCE, ACE, seg, coords = data
        print(coords[:, 0, :], coords[:, 0, 0])
        imgs, seg = TestAug.augment(imgs=[NCE, ACE], seg=seg)
        NCE, ACE = imgs
        coords = tf.cast(coords, tf.int32)
        c1 = coords[0][1]
        c2 = coords[1][1]
        NCE1, NCE2 = NCE[0, ...], NCE[1, ...]
        ACE1, ACE2 = ACE[0, ...], ACE[1, ...]
        seg1, seg2 = seg[0, ...], seg[1, ...]
        # NCE1 = NCE[0, int(c1[0]) - 100:int(c1[0]) + 100, int(c1[1]) - 100:int(c1[1]) + 100, :, :]
        # NCE2 = NCE[1, int(c2[0]) - 100:int(c2[0]) + 100, int(c2[1]) - 100:int(c2[1]) + 100, :, :]
        # ACE1 = ACE[0, int(c1[0]) - 100:int(c1[0]) + 100, int(c1[1]) - 100:int(c1[1]) + 100, :, :]
        # ACE2 = ACE[1, int(c2[0]) - 100:int(c2[0]) + 100, int(c2[1]) - 100:int(c2[1]) + 100, :, :]
        # seg1 = seg[0, int(c1[0]) - 100:int(c1[0]) + 100, int(c1[1]) - 100:int(c1[1]) + 100, :, :]
        # seg2 = seg[1, int(c2[0]) - 100:int(c2[0]) + 100, int(c2[1]) - 100:int(c2[1]) + 100, :, :]

        plt.subplot(2, 3, 1)
        plt.imshow(NCE1[:, :, 0, 0])
        plt.subplot(2, 3, 4)
        plt.imshow(NCE2[:, :, 0, 0])
        
        plt.subplot(2, 3, 2)
        plt.imshow(ACE1[:, :, 0, 0])
        plt.subplot(2, 3, 5)
        plt.imshow(ACE2[:, :, 0, 0])

        plt.subplot(2, 3, 3)
        plt.imshow(seg1[:, :, 0, 0])
        plt.subplot(2, 3, 6)
        plt.imshow(seg2[:, :, 0, 0])

        plt.show()