import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------------------------------------------------
""" ImgLoader class: data_generator method for use with tf.data.Dataset.from_generator """

class BaseImgLoader(ABC):
    def __init__(self, config: dict, dataset_type: str):
        # Expects at least two sub-folders within data folder e.g. "AC", "VC, "HQ"
        img_path = f"{config['data_path']}/Images"
        seg_path = f"{config['data_path']}/Segmentations"
        self.sub_folders = [f for f in os.listdir(img_path) if os.path.isdir(f"{img_path}/{f}")]
        self.seg_folders = [f for f in os.listdir(seg_path) if os.path.isdir(f"{img_path}/{f}")]

        if len(self.sub_folders) == 0:
            print("==================================================")
            print("Assuming unpaired dataset")
            self._img_paths = img_path
            self._seg_paths = seg_path

        else:
            self._img_paths = {key: f"{img_path}/{key}" for key in self.sub_folders}
            self._seg_paths = {key: f"{seg_path}/{key}" for key in self.seg_folders}

        self._dataset_type = dataset_type
        self.config = config
        self.down_sample = config["down_sample"]

        if config["times"] is not None:
            self._json = json.load(open(f"{config['data_path']}/{config['times']}", 'r'))
        
        else:
            self._json = None

    def example_images(self):
        if len(self._ex_segs) > 0:
            if self._json is not None:
                return self._normalise(self._ex_sources), self._normalise(self._ex_targets), self._ex_segs, self._ex_source_times, self._ex_target_times
            else:
                return self._normalise(self._ex_sources), self._normalise(self._ex_targets), self._ex_segs

        else:
            if self._json is not None:
                return self._normalise(self._ex_sources), self._normalise(self._ex_targets), self._ex_source_times, self._ex_target_times
            else:
                return self._normalise(self._ex_sources), self._normalise(self._ex_targets)
    
    def train_val_split(self, seed: int = 5) -> None:
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

        if self.config["fold"] > self.config["cv_folds"] - 1:
            raise ValueError(f"Fold number {self.config['fold']} of {self.config['cv_folds']} folds")

        np.random.seed(seed)
        N = len(self._unique_ids)
        np.random.shuffle(self._unique_ids)

        # Split into folds by subject
        if self.config["cv_folds"] > 1:
            if seed == None:
                self._unique_ids.sort()

            num_in_fold = N // self.config["cv_folds"]

            if self._dataset_type == "training":
                fold_ids = self._unique_ids[0:self.config["fold"] * num_in_fold] + self._unique_ids[(self.config["fold"] + 1) * num_in_fold:]
            elif self._dataset_type == "validation":
                fold_ids = self._unique_ids[self.config["fold"] * num_in_fold:(self.config["fold"] + 1) * num_in_fold]
            else:
                raise ValueError("Select 'training' or 'validation'")

            self._fold_targets = []
            self._fold_sources = []
            self._fold_targets = sorted([img for img in self._targets if img[0:4] in fold_ids])
            self._fold_sources = sorted([img for img in self._sources if img[0:4] in fold_ids])
            self._fold_segs = sorted([seg for seg in self._segs if seg[0:4] in fold_ids])
        
        elif self.config["cv_folds"] == 1:
            self._fold_targets = self._targets
            self._fold_sources = self._sources
            self._fold_segs = self._segs
        
        else:
            raise ValueError("Number of folds must be > 0")

        example_idx = np.random.randint(0, len(self._fold_sources), self.config["num_examples"])
        ex_sources_list = list(np.array([self._fold_sources]).squeeze()[example_idx])

        if len(self.sub_folders) == 0:
            ex_targets_list = [np.random.choice([t[0:11] + s[-8:] for t in self._fold_targets if s[0:6] in t and t[0:11] + s[-8:] not in s]) for s in ex_sources_list]
            self._ex_sources = np.stack([np.load(f"{self._img_paths}/{img}") for img in ex_sources_list], axis=0)
            self._ex_targets = np.stack([np.load(f"{self._img_paths}/{img}") for img in ex_targets_list], axis=0)
        else:
            ex_targets_list = list(np.array([self._fold_targets]).squeeze()[example_idx])
            self._ex_sources = np.stack([np.load(f"{self._img_paths[img[6:8]]}/{img}") for img in ex_sources_list], axis=0)
            self._ex_targets = np.stack([np.load(f"{self._img_paths[img[6:8]]}/{img}") for img in ex_targets_list], axis=0)

        self._ex_sources = self._ex_sources[:, ::self.down_sample, ::self.down_sample, :, np.newaxis].astype("float32")
        self._ex_targets = self._ex_targets[:, ::self.down_sample, ::self.down_sample, :, np.newaxis].astype("float32")

        if len(self.config["segs"]) > 0 and len(self.sub_folders) == 0:
            candidate_segs = [glob.glob(f"{self._seg_paths}/{img[0:6]}AC*{img[-8:]}")[0] for img in ex_targets_list]
            self._ex_segs = np.stack([np.load(seg) for seg in candidate_segs], axis=0)
            self._ex_segs = self._ex_segs[:, ::self.down_sample, ::self.down_sample, :, np.newaxis].astype("float32")

        elif len(self.config["segs"]) > 0 and len(self.sub_folders) > 0:
            self._ex_segs = np.stack([np.load(f"{self._seg_paths[img[6:8]]}/{img}") for img in ex_targets_list], axis=0)
            self._ex_segs = self._ex_segs[:, ::self.down_sample, ::self.down_sample, :, np.newaxis].astype("float32")

        else:
            self._ex_segs = []

        if self._json is not None:
            self._ex_source_times = np.stack([self._json[name[:-8] + ".nrrd"] for name in ex_sources_list], axis=0).astype("float32")
            self._ex_target_times = np.stack([self._json[name[:-8] + ".nrrd"] for name in ex_targets_list], axis=0).astype("float32")

        else:
            self._ex_source_times = []
            self._ex_target_times = []

        np.random.seed()
        
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
    
    def set_normalisation(self, param_1: float = None, param_2: float = None):
        # Mean -281.528, std = 261.552
        # Min -500, max = 22451
        self.norm_type = self.config["norm_type"]

        # Override if custom parameters passed
        if param_1 is not None and param_2 is not None:
            self.param_1 = param_1
            self.param_2 = param_2

        # Otherwise, use parameters provided in config yaml
        elif self.config["norm_param_1"] is not None and self.config["norm_param_1"] is not None:
            self.param_1 = self.config["norm_param_1"]
            self.param_2 = self.config["norm_param_2"]

        # Otherwise, calculate parameters
        else:
            # If mean and std of data not available, we get rolling averages
            if self.norm_type == "meanstd" or self.norm_type == "std":
                mean = 0
                std = 0

                for img in self._targets + self._sources:
                    im = np.load(f"{self._img_paths[img[6:8]]}/{img}")
                    mean = 0.99 * mean + 0.01 * im.mean()
                    std = 0.99 * std + 0.01 * im.std()
                
                self.param_1 = mean
                self.param_2 = std

            # If min and max not available, we get min and max of whole dataset
            elif self.norm_type == "minmax":
                min_val = 2048
                max_val = -2048

                for img in self._targets + self._sources:
                    im = np.load(f"{self._img_paths[img[6:8]]}/{img}")
                    min_val = np.min([min_val, im.min()])
                    max_val = np.max([max_val, im.max()])
                
                self.param_1 = min_val
                self.param_2 = max_val
        
            else:
                raise ValueError("Choose meanstd or minmax")

        print("==================================================")
        print(f"{self.norm_type} normalisation: mean/min {self.param_1}, std/max {self.param_2}")

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
    
    def un_normalise(self, img):
        if self.norm_type == "meanstd":
            return img * self.param_2 + self.param_1
        elif self.norm_type == "std":
            return img * self.param_2
        else:
            return img * (self.param_2 - self.param_1) + self.param_1

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

            if len(self.sub_folders) == 0:
                target = np.load(f"{self._img_paths}/{target_name}")
                source = np.load(f"{self._img_paths}/{source_name}")
            else:
                target = np.load(f"{self._img_paths[target_name[6:8]]}/{target_name}")
                source = np.load(f"{self._img_paths[source_name[6:8]]}/{source_name}")

            target = target[::self.down_sample, ::self.down_sample, :, np.newaxis]
            source = source[::self.down_sample, ::self.down_sample, :, np.newaxis]
            target = self._normalise(target)
            source = self._normalise(source)

            if self._json is not None:
                source_time = self._json[names["source"][:-8] + ".nrrd"]
                target_time = self._json[names["target"][:-8] + ".nrrd"]

            # TODO: allow using different seg channels
            if len(self._fold_segs) > 0:
                if len(self.sub_folders) == 0:
                    candidate_segs = glob.glob(f"{self._seg_paths}/{target_name[0:6]}AC*{target_name[-8:]}")
                    assert len(candidate_segs) == 1, candidate_segs
                    seg = np.load(candidate_segs[0]).astype("float32")
                    seg = seg[::self.down_sample, ::self.down_sample, :, np.newaxis]
                    seg[seg > 1] = 1
                    # TODO: return index

                else:
                    seg = np.load(f"{self._seg_paths[target_name[6:8]]}/{target_name}").astype("float32")
                    seg = seg[::self.down_sample, ::self.down_sample, :, np.newaxis]
                    seg[seg > 1] = 1

                if self._json is not None:
                    yield (source, target, seg, source_time, target_time)
                else:
                    yield (source, target, seg)

            else:
                if self._json is not None:
                    yield (source, target, source_time, target_time)
                else:
                    yield (source, target)

            i += 1

#-------------------------------------------------------------------------
""" Data loader for one to one source-target pairings """

class PairedLoader(BaseImgLoader):
    def __init__(self, config: dict, dataset_type: str):
        super().__init__(config, dataset_type)

        # Expects list of targets and sources e.g. ["AC", "VC"], ["HQ"]
        self._targets = []
        self._sources = []
        self._segs = []

        if len(config["target"]) > 0:
            for key in config["target"]:
                self._targets += os.listdir(self._img_paths[key])

        elif len(config["target"]) == 0:
            for key in self.sub_folders:
                self._targets += os.listdir(self._img_paths[key])

        if len(config["source"]) > 0:
            for key in config["source"]:
                self._sources += os.listdir(self._img_paths[key])

        elif len(config["source"]) == 0:
            for key in self.sub_folders:
                self._sources += os.listdir(self._img_paths[key])
        
        if len(config["segs"]) > 0:
            for key in config["segs"]:
                self._segs += os.listdir(self._seg_paths[key])

        if len(self._targets) == 0 or len(self._sources) == 0:
            raise FileNotFoundError(f"No data found: {len(self._targets)} targets, {len(self._sources)} sources")

        print("==================================================")
        print(f"Data: {len(self._targets)} targets, {len(self._sources)} sources, {len(self._segs)} segmentations")
        print(f"Using paired loader for {self._dataset_type}")

        super().train_val_split()

        self._subject_targets = {k: [img for img in v if img[6:8] in self.config["target"]] for k, v in self._subject_imgs.items()}
        self._subject_sources = {k: [img for img in v if img[6:8] in self.config["source"]] for k, v in self._subject_imgs.items()}
    
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

        # Optional list of targets and sources e.g. ["AC", "VC"], ["HQ"]
        self._targets = []
        self._sources = []
        self._segs = []

        if len(config["target"]) > 0:
            for key in config["target"]:
                self._targets += [t for t in os.listdir(self._img_paths) if key in t]

        elif len(config["target"]) == 0:
            self._targets += os.listdir(self._img_paths)

        if len(config["source"]) > 0:
            for key in config["source"]:
                self._sources += [s for s in os.listdir(self._img_paths) if key in s]

        elif len(config["source"]) == 0:
            self._sources += os.listdir(self._img_paths)
        
        if len(config["segs"]) > 0:
            self._segs += os.listdir(self._seg_paths)

        print("==================================================")
        print(f"Data: {len(self._targets)} targets, {len(self._sources)} sources, {len(self._segs)} segmentations")
        print(f"Using unpaired loader for {self._dataset_type}")

        super().train_val_split()

        if len(self.config["target"]) > 0:
            self._subject_targets = {k: [img for img in v if img[6:8] in self.config["target"]] for k, v in self._subject_imgs.items()}
        else:
            self._subject_targets = None

    @property
    def subject_imgs(self):
        """ Return list of images indexed by procedure """
        return self._subject_imgs

    def img_pairer(self, source: object, direction: str = None) -> dict:
        # TODO: add forwards/backwards sampling, return idx
        if self._subject_targets is None:
            target_candidates = list(self._subject_imgs[source[0:6]])
        else:
            target_candidates = list(self._subject_targets[source[0:6]])

        try:
            target_candidates.remove(source[0:-8])
        except ValueError:
            pass

        assert len(target_candidates) > 0, source
        target_stem = target_candidates[np.random.randint(len(target_candidates))]
        target = f"{target_stem}_{source[-7:]}"

        return {"target": target, "source": source}


#----------------------------------------------------------------------------------------------------------------------------------------------------
 
if __name__ == "__main__":
    import yaml

    """ Routine for visually testing dataloader """

    test_config = yaml.load(open("syntheticcontrast_v02/utils/test_config.yml", 'r'), Loader=yaml.FullLoader)

    TestLoader = UnpairedLoader(config=test_config["data"], dataset_type="training")
    _, _ = TestLoader.set_normalisation()

    output_types = ["float32", "float32"]

    if len(test_config["data"]["segs"]) > 0:
        output_types += ["float32"]
    
    if test_config["data"]["times"] is not None:
        output_types += ["float32", "float32"]
    
    train_ds = tf.data.Dataset.from_generator(TestLoader.data_generator, output_types=tuple(output_types))

    for data in train_ds.batch(2).take(2):
        if len(test_config["data"]["segs"]) > 0:
            source, target, seg, source_time, target_time = data
        else:
            source, target, source_time, target_time = data

        source = TestLoader.un_normalise(source)
        target = TestLoader.un_normalise(target)

        plt.subplot(3, 2, 1)
        plt.imshow(source[0, :, :, 0, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(source_time[0].numpy())

        plt.subplot(3, 2, 2)
        plt.imshow(source[1, :, :, 0, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(source_time[1].numpy())

        plt.subplot(3, 2, 3)
        plt.imshow(target[0, :, :, 0, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(target_time[0].numpy())

        plt.subplot(3, 2, 4)
        plt.imshow(target[1, :, :, 0, 0], cmap="gray", vmin=-150, vmax=250)
        plt.axis("off")

        if test_config["data"]["times"] is not None:
            plt.title(target_time[1].numpy())

        if len(test_config["data"]["segs"]) > 0:
            plt.subplot(3, 2, 5)
            plt.imshow(seg[0, :, :, 0, 0])
            plt.axis("off")
            plt.subplot(3, 2, 6)
            plt.imshow(seg[1, :, :, 0, 0])
            plt.axis("off")

        plt.show()

    if len(test_config["data"]["segs"]) > 0:
        if test_config["data"]["times"] is not None:
            source, target, seg, source_time, target_time = TestLoader.example_images()
        else:
            source, target, seg = TestLoader.example_images()
    else:
        if test_config["data"]["times"] is not None:
            source, target, source_time, target_time = TestLoader.example_images()
        else:
            source, target = TestLoader.example_images()

    source = TestLoader.un_normalise(source)
    target = TestLoader.un_normalise(target)
    
    fig, axs = plt.subplots(target.shape[0], 3)

    for i in range(target.shape[0]):
        axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 0].axis("off")

        if test_config["data"]["times"] is not None:
            axs[i, 0].set_title(source_time[i])

        axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
        axs[i, 1].axis("off")

        if test_config["data"]["times"] is not None:
            axs[i, 1].set_title(target_time[i])

        if len(test_config["data"]["segs"]) > 0:
            axs[i, 2].imshow(seg[i, :, :, 11, 0])
            axs[i, 2].axis("off")
    
    plt.show()
