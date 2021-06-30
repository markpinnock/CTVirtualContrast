import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time

from abc import ABC, abstractmethod

np.set_printoptions(suppress=True)


#-------------------------------------------------------------------------
""" Base training loop class """

class BaseTrainingLoop(ABC):

    def __init__(self, Model: object, dataset: object, val_generator: object, config: dict):

        self.Model = Model
        self.train_ds, self.val_ds = dataset
        self.val_generator = val_generator
        self._config = config
        self.EPOCHS = config["EXPT"]["EPOCHS"]
        self.IMAGE_SAVE_PATH = f"{config['EXPT']['SAVE_PATH']}images/{config['EXPT']['MODEL']}/{config['EXPT']['EXPT_NAME']}/"
        self.MODEL_SAVE_PATH = f"{config['EXPT']['SAVE_PATH']}models/{config['EXPT']['MODEL']}/{config['EXPT']['EXPT_NAME']}/"
        self.LOG_SAVE_PATH = f"{config['EXPT']['SAVE_PATH']}logs/{config['EXPT']['MODEL']}/{config['EXPT']['EXPT_NAME']}/"
        self.SAVE_EVERY = config["EXPT"]["SAVE_EVERY"]

        if not os.path.exists(self.IMAGE_SAVE_PATH): os.makedirs(self.IMAGE_SAVE_PATH)
        if not os.path.exists(self.MODEL_SAVE_PATH): os.makedirs(self.MODEL_SAVE_PATH)
        if not os.path.exists(self.LOG_SAVE_PATH): os.makedirs(self.LOG_SAVE_PATH)

        self.ds_train, self.ds_val = dataset
        
    @abstractmethod
    def training_loop(self):
        """ Main training loop for models """

        raise NotImplementedError

    def save_images(self, target, source, pred, epoch=None, tuning_path=None):
        """ Saves sample of images """

        fig, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray")
            axs[i, 1].axis("off")
            axs[i, 3].imshow(np.abs(target[i, :, :, 11, 0] - source[i, :, :, 11, 0]), cmap="hot")
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, :, :, 11, 0], cmap="gray")
            axs[i, 2].axis("off")
            axs[i, 4].imshow(np.abs(target[i, :, :, 11, 0] - pred[i, :, :, 11, 0]), cmap="hot")
            axs[i, 4].axis("off")

        plt.tight_layout()

        if tuning_path:
            plt.savefig(f"{tuning_path}.png", dpi=250)
        else:
            plt.savefig(f"{self.IMAGE_SAVE_PATH}{epoch}.png", dpi=250)

        plt.close()

    @abstractmethod
    def save_results(self):
        """ Saves json of results """

        json.dump(self.results, open(f"{self.LOG_SAVE_PATH}results.json", 'w'), indent=4)
    
    @property
    def results(self):
        """ Potentially avoids shallow copy problem """
        return self._results
        # return {key: val for key, val in self._results.items()}
    
    @property
    def config(self):
        """ Potentially avoids shallow copy problem """
        return self._config
        # return {key: val for key, val in self._config.items()}

#-------------------------------------------------------------------------
""" UNet training loop - inherits from BaseTrainingLoop """

class TrainingLoopUNet(BaseTrainingLoop):

    def __init__(self, Model, dataset, config):
        super().__init__(Model, dataset, config)

    def training_loop(self, verbose=1):
        """ Main training loop for UNet """

        self._results = {}
        self._results["train_metric"] = {"global": [], "focal": [], "weights": []}
        self._results["val_metric"] = {"global": [], "focal": [], "weights": []}
        self._results["config"] = self._config
        self._results["epochs"] = []
        self._results["time"] = 0

        start_time = time.time()

        for epoch in range(self.EPOCHS):
            self._results["epochs"].append(epoch + 1)
            self.Model.metric.reset_states()

            for data in self.ds_train:
                self.Model.train_step(data)

            self._results["train_metric"]["global"].append(float(self.Model.metric.result()[0]))
            self._results["train_metric"]["focal"].append(float(self.Model.metric.result()[1]))
            self._results["train_metric"]["weights"].append(float(self.Model.metric.result()[2]))
            
            if verbose:
                print(f"Train epoch {epoch + 1}, loss [global, focal, weights]: {self.Model.metric.result()}")

            if self.config["EXPT"]["CV_FOLDS"]:
                self.Model.metric.reset_states()

                for data in self.ds_val:
                    self.Model.val_step(data)

                self._results["val_metric"]["global"].append(float(self.Model.metric.result()[0]))
                self._results["val_metric"]["focal"].append(float(self.Model.metric.result()[1]))
                self._results["val_metric"]["weights"].append(float(self.Model.metric.result()[2]))

                if verbose:
                    print(f"Val epoch {epoch + 1}, loss [global, focal, weights]: {self.Model.metric.result()}")

            if (epoch + 1) % self.SAVE_EVERY == 0 and verbose:
                if self.config["EXPT"]["CROP"]:
                    self.save_images_ROI(epoch + 1)
                else:
                    self.save_images(epoch + 1)
            
            self.Model.save_weights(f"{self.MODEL_SAVE_PATH}/{self.config['EXPT']['EXPT_NAME']}")
            self.save_results()

        self._results["time"] = (time.time() - start_time) / 3600

        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")
            self.Model.save_weights(f"{self.MODEL_SAVE_PATH}/{self.config['EXPT']['EXPT_NAME']}")

    def save_images(self, epoch=None, tuning_path=None):
        """ Saves sample of images """

        NCE, ACE, _, _ = next(iter(self.ds_val))
        NCE, ACE = NCE[0:4, ...].numpy(), ACE[0:4, ...].numpy()
        pred = self.Model(NCE, training=False).numpy()
        super().save_images(NCE, ACE, pred, epoch, tuning_path)
    
    def save_images_ROI(self, epoch=None, tuning_path=None):
        """ Saves sample of cropped images """

        ds_iter = iter(self.ds_val)
        _, _, _, _ = next(ds_iter)
        NCE, ACE, seg, coords = next(ds_iter)
        NCEs, ACEs, pred = [], [], []

        for i in range(3):
            rNCE, rACE, _ = self.Model.crop_ROI(NCE, ACE, seg, coords[:, i, :])
            NCEs.append(rNCE[0, ...])
            ACEs.append(rACE[0, ...])
        
        NCEs = tf.stack(NCEs, axis=0)
        ACEs = tf.stack(ACEs, axis=0)

        pred = self.Model(NCEs, training=False).numpy()
        super().save_images(NCEs.numpy(), ACEs.numpy(), pred, epoch, tuning_path)
    
    def save_results(self, tuning_path=None):
        """ Saves json of results and saves loss curves """

        plt.figure()
        plt.plot(self.results["epochs"], self.results["train_metric"]["global"], 'k--', label="Train global")
        plt.plot(self.results["epochs"], self.results["train_metric"]["focal"], 'r--', label="Train focal")

        if self.config["EXPT"]["CV_FOLDS"]:
            plt.plot(self.results["epochs"], self.results["val_metric"]["global"], 'k-', label="Val global")
            plt.plot(self.results["epochs"], self.results["val_metric"]["focal"], 'r-', label="Val focal")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Losses")
        plt.legend()

        plt.tight_layout()
        
        if tuning_path:
            plt.savefig(tuning_path)
        else:
            plt.savefig(f"{self.LOG_SAVE_PATH}losses.png")
            super().save_results()

        plt.close()

#-------------------------------------------------------------------------
""" GAN training loop - inherits from BaseTrainingLoop """

class TrainingLoopGAN(BaseTrainingLoop):

    def __init__(self, Model, dataset, val_generator, config):
        super().__init__(Model, dataset, val_generator, config)

    def training_loop(self, verbose=1):
        """ Main training loop for GAN """

        self._results = {}
        self._results["g_metric"] = []
        self._results["d_metric"] = []
        self._results["train_L1"] = []
        self._results["val_L1"] = []
        self._results["config"] = self._config
        self._results["epochs"] = []
        self._results["time"] = 0

        start_time = time.time()
        best_L1 = 1e6

        for epoch in range(self.EPOCHS):
            self._results["epochs"].append(epoch + 1)

            # for key, value in model.generator_metrics.items():
            self.Model.g_metric.reset_states()
            self.Model.train_L1_metric.reset_states()

            # for value in model.discriminator_metrics.values():
            self.Model.d_metric.reset_states()

            for data in self.ds_train:
                self.Model.train_step(data)

            # for key, value in model.generator_metrics.items():
            self._results["g_metric"].append(float(self.Model.g_metric.result()))
            self._results["train_L1"].append(float(self.Model.train_L1_metric.result()))
            
            # for key, value in model.discriminator_metrics.items():
            self._results["d_metric"].append(float(self.Model.d_metric.result()))

            # for key in model.discriminator_metrics.keys():
            if verbose:
                print(f"Train epoch {epoch + 1}, G: {self.Model.g_metric.result():.4f} D: {self.Model.d_metric.result():.4f}, L1: {self.Model.train_L1_metric.result()}")

            if self.config["EXPT"]["CV_FOLDS"] > 1:
                self.Model.val_L1_metric.reset_states()

                for data in self.ds_val:
                    self.Model.val_step(data)
                
                self._results["val_L1"].append(float(self.Model.val_L1_metric.result()))

                if verbose:
                    print(f"Val epoch {epoch + 1}, L1: {self.Model.val_L1_metric.result()}")

            if (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_images(epoch + 1)

            if self.Model.val_L1_metric.result() < best_L1 and not self.config["EXPT"]["VERBOSE"]:
                self.Model.save_weights(f"{self.MODEL_SAVE_PATH}{self.config['EXPT']['EXPT_NAME']}")
                best_L1 = self.Model.val_L1_metric.result()
                super().save_results()

        self._results["time"] = (time.time() - start_time) / 3600
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")
    
    def save_images(self, epoch=None, tuning_path=None):
        """ Saves sample of images """

        source, target = self.val_generator.example_images()
        pred = self.Model.Generator(source, training=False).numpy()
        super().save_images(source, target, pred, epoch, tuning_path)

    def save_results(self, tuning_path=None):
        """ Saves json of results and saves loss curves """

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(self.results["epochs"], self.results["g_metric"], 'k', label="G")
        plt.plot(self.results["epochs"], self.results["d_metric"], 'r', label="D")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Losses")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.results["epochs"], self.results["train_L1"], 'k-', label="Train L1")

        if self.config["EXPT"]["CV_FOLDS"]:
            plt.plot(self.results["epochs"], self.results["val_L1"], 'r-', label="Val L1")

        plt.xlabel("Epochs")
        plt.ylabel("L1")
        plt.title("Metrics")
        plt.legend()

        if tuning_path:
            plt.savefig(tuning_path)
        else:
            plt.savefig(f"{self.LOG_SAVE_PATH}losses.png")
            super().save_results()
        
        plt.close()
