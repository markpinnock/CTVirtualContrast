import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" ImgLoader class: data_generator method for use with tf.data.Dataset.from_generator """

class BaseImgLoader(ABC):
    def __init__(self, config: dict, dataset_type: str):
        # Expects at least two sub-folders within data folder e.g. "AC", "VC, "HQ"
        file_path = config["DATA_PATH"]
        sub_folders = [f for f in os.listdir(file_path) if os.path.isdir(f"{file_path}/{f}")]

        if len(sub_folders) == 0:
            raise FileNotFoundError("No subfolders found")

        self._data_paths = {key: f"{file_path}/{key}" for key in sub_folders}
        self._dataset_type = dataset_type
        self.config = config
        self.down_sample = config["DOWN_SAMP"]

        if config["DATA"]["JSON"] != "":
            self._json = json.load(open(f"{file_path}/{config['DATA']['JSON']}", 'r'))

        # Expects list of targets and sources e.g. ["AC", "VC"], ["HQ"], leave blank for unpaired
        self._targets = []
        self._sources = []

        if len(config["DATA"]["TARGET"]) > 0:
            for key in config["DATA"]["TARGET"]:
                self._targets += os.listdir(self._data_paths[key])

        elif len(config["DATA"]["TARGET"]) == 0:
            for key in sub_folders:
                self._targets += os.listdir(self._data_paths[key])

        if len(config["DATA"]["SOURCE"]) > 0:
            for key in config["DATA"]["SOURCE"]:
                self._sources += os.listdir(self._data_paths[key])

        elif len(config["DATA"]["SOURCE"]) == 0:
            for key in sub_folders:
                self._sources += os.listdir(self._data_paths[key])

        if len(self._targets) == 0 or len(self._sources) == 0:
            raise FileNotFoundError(f"No data found: {len(self._targets)} targets, {len(self._sources)} sources")

        print("==================================================")
        print(f"Data: {len(self._targets)} targets, {len(self._sources)} sources")

        # Get unique subject IDs for subject-level train/val split
        self._unique_ids = []

        for img_id in self._targets:
            if img_id[0:4] not in self._unique_ids:
                self._unique_ids.append(img_id[0:4])

        self._unique_ids.sort()
        self._subject_imgs = {}

        # Need procedure IDs (as poss. >1 per subject) to build ordered index of subjects' images
        self._subject_imgs = {}

        for img_id in self._targets + self._sources:
            if img_id[0:6] not in self._subject_imgs.keys():
                self._subject_imgs[img_id[0:6]] = []
            if img_id[:-8] not in self._subject_imgs[img_id[0:6]]:
                self._subject_imgs[img_id[0:6]].append(img_id[:-8])

        for key in self._subject_imgs.keys():
            self._subject_imgs[key] = sorted(self._subject_imgs[key], key=lambda x: int(x[-3:]))

    def example_images(self):
        raise NotImplementedError
    
    def train_val_split(self, seed: int = 5) -> None:
        if self.config["FOLD"] > self.config["CV_FOLDS"] - 1:
            raise ValueError(f"Fold number {self.config['FOLD']} of {self.config['CV_FOLDS']} folds")

        np.random.seed(seed)
        N = len(self._unique_ids)
        np.random.shuffle(self._unique_ids)

        # Split into folds by subject
        if self.config["CV_FOLDS"] > 1:
            if seed == None:
                self._unique_ids.sort()

            num_in_fold = N // self.config["CV_FOLDS"]

            if self._dataset_type == "training":
                fold_ids = self._unique_ids[0:self.config["FOLD"] * num_in_fold] + self._unique_ids[(self.config["FOLD"] + 1) * num_in_fold:]
            elif self._dataset_type == "validation":
                fold_ids = self._unique_ids[self.config["FOLD"] * num_in_fold:(self.config["FOLD"] + 1) * num_in_fold]
            else:
                raise ValueError("Select 'training' or 'validation'")

            self._fold_targets = []
            self._fold_sources = []
            self._fold_targets = [img for img in self._targets if img[0:4] in fold_ids]
            self._fold_sources = [img for img in self._sources if img[0:4] in fold_ids]

            np.random.seed()
        
        elif self.config["CV_FOLDS"] == 1:
            self._fold_targets = self._targets
            self._fold_sources = self._sources
        
        else:
            raise ValueError("Number of folds must be > 0")

        assert len(self._fold_targets) == len(self._fold_sources)
        
        print(f"{len(self._fold_targets)} of {len(self._targets)} examples in {self._dataset_type} folds")

    @property
    def unique_ids(self) -> list:
        return self._unique_ids
    
    @property
    def data(self) -> dict:
        """ Return list of all images """
        return {"targets": self._targets, "sources": self._sources}
    
    @property
    def fold_data(self) -> dict:
        """ Return list of all images in training or validation fold """
        return {"targets": self._fold_targets, "sources": self._fold_sources}
    
    @property
    def subject_imgs(self):
        raise NotImplementedError
    
    def set_normalisation(self, norm_type, param_1=None, param_2=None):
        # Mean -281.528, std = 261.552
        if param_1 != None and param_2 != None:
            self.param_1 = param_1
            self.param_2 = param_2

        else:
            self.norm_type = norm_type

            # If mean and std of data not available, we get rolling averages
            if norm_type == "meanstd" or norm_type == "std":
                mean = 0
                std = 0

                for img in self._targets + self._sources:
                    im = np.load(f"{self._data_paths[img[6:8]]}/{img}")
                    mean = 0.99 * mean + 0.01 * im.mean()
                    std = 0.99 * std + 0.01 * im.std()
                
                self.param_1 = mean
                self.param_2 = std

            # If min and max not available, we get min and max of whole dataset
            elif norm_type == "minmax":
                min_val = 2048
                max_val = -2048

                for img in self._targets + self._sources:
                    im = np.load(f"{self._data_paths[img[6:8]]}/{img}")
                    min_val = np.min([min_val, im.min()])
                    max_val = np.max([max_val, im.max()])
                
                self.param_1 = min_val
                self.param_2 = max_val
        
            else:
                raise ValueError("Choose meanstd or minmax")

        print(f"{norm_type} normalisation: mean/min {self.param_1}, std/max {self.param_2}")
        print("==================================================")

        return self.param_1, self.param_2
    
    @property
    def norm_params(self):
        """ Return mean/std or min/max parameters """
        return (self.param_1, self.param_2)
    
    @abstractmethod
    def img_pairer(self):
        raise NotImplementedError
    
    def _normalise(self, img):
        if self.norm_type == "meanstd":
            return (img - self.param_1) / self.param_2
        elif self.norm_type == "std":
            return img / self.param_2
        else:
            return (img - self.param_1) / (self.param_2 - self.param_1)

    def data_generator(self):
        if self._dataset_type == "training":
            np.random.shuffle(self._fold_sources)

        N = len(self._fold_sources)
        i = 0

        # Pair source and target images
        while i < N:
            source_name = self._fold_sources[i]
            names = self.img_pairer(source_name)
            target_name = names["target"]
            source_name = names["source"]
            target = np.load(f"{self._data_paths[target_name[6:8]]}/{target_name}")
            source = np.load(f"{self._data_paths[source_name[6:8]]}/{source_name}")

            target = target[::self.down_sample, ::self.down_sample, :, np.newaxis]
            source = source[::self.down_sample, ::self.down_sample, :, np.newaxis]
            target = self._normalise(target)
            source = self._normalise(source)

            # TODO: return index
            yield (target, source)

            i += 1

#-------------------------------------------------------------------------
""" Data loader for one to one source-target pairings """

class PairedLoader(BaseImgLoader):
    def __init__(self, config: dict, dataset_type: str):
        super().__init__(config, dataset_type)
        print(f"Using paired loader for {self._dataset_type}")
        super().train_val_split()
        self._subject_targets = {k: [img for img in v if img[6:8] in self.config["DATA"]["TARGET"]] for k, v in self._subject_imgs.items()}
        self._subject_sources = {k: [img for img in v if img[6:8] in self.config["DATA"]["SOURCE"]] for k, v in self._subject_imgs.items()}
    
    @property
    def subject_imgs(self):
        """ Return list of images indexed by procedure """
        return {"targets": self._subject_targets, "sources": self._subject_sources}

    def img_pairer(self, source: str) -> dict:
        # TODO: return idx
        # Get potential target candidates matching source (where target and source specified)
        target_candidates = self._subject_targets[source[0:6]]
        assert len(target_candidates) > 0, source
        target_stem = target_candidates[np.random.randint(len(target_candidates))]
        target = f"{target_stem}_{source[-7:]}"

        return {"target": target, "source": source}

#-------------------------------------------------------------------------
""" Data loader for unpaired images """

class UnpairedLoader(BaseImgLoader):
    def __init__(self, config: dict, dataset_type: str):
        super().__init__(config, dataset_type)
        print(f"Using unpaired loader for {self._dataset_type}")
        super().train_val_split()

    @property
    def subject_imgs(self):
        """ Return list of images indexed by procedure """
        return self._subject_imgs

    def img_pairer(self, source: object, direction: str = None) -> dict:
        # TODO: add forwards/backwards sampling, return idx
        target_candidates = self._subject_imgs[source[0:6]]
        target_candidates.remove(source[0:-8])
        assert len(target_candidates) > 0, source
        target_stem = target_candidates[np.random.randint(len(target_candidates))]
        target = f"{target_stem}_{source[-7:]}"

        return {"target": target, "source": source}

#-------------------------------------------------------------------------
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