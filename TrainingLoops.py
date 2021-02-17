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
""" Function to print summary of model e.g. layers, feature map dims """

def print_model_summary(model, config, model_type):
    print("===================================")
    print(model.name)
    print("===================================")

    weight_dict = {}
    total_weights = 0
    if model_type == "G":
        in_ch = 1
    else:
        in_ch = config["HYPERPARAMS"]["D_IN_CH"]

    xy_dims = np.array(config["EXPT"]["IMG_DIMS"])[0:2] // config["EXPT"]["DOWN_SAMP"]
    z_dims = np.array(config["EXPT"]["IMG_DIMS"])[2]
    x = np.zeros([1, xy_dims[0], xy_dims[1], z_dims, in_ch], dtype=np.float32)
    output_shapes = model(x, test=True)

    for layer in model.layers:
        weight_dict[layer.name] = []

        for weight in layer.trainable_weights:
            total_weights += np.prod(weight.shape.as_list())
            weight_dict[layer.name].append(weight.shape.as_list())
    
    assert len(output_shapes) == len(weight_dict), "Weight list and output shape list do not match"

    for idx, layer in enumerate(model.layers):
        print(f"{layer.name}: {weight_dict[layer.name]} {output_shapes[idx]}")

    print("===================================")
    print(f"Total weights: {total_weights}")
    print("===================================")

#-------------------------------------------------------------------------
""" Base training loop class """

class BaseTrainingLoop(ABC):

    def __init__(self, Model: object, dataset: object, config: dict):

        self.Model = Model
        self.train_ds, self.val_ds = dataset
        self.config = config
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

    def save_images(self, ACE, NCE, pred, epoch):
        """ Saves sample of images """

        fig, axs = plt.subplots(ACE.shape[0], 4)

        for i in range(ACE.shape[0]):
            axs[i, 0].imshow(NCE[i, :, :, 11, 0].T, cmap="gray")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(ACE[i, :, :, 11, 0].T, cmap="gray")
            axs[i, 1].axis("off")
            axs[i, 2].imshow(pred[i, :, :, 11, 0].T, cmap="gray")
            axs[i, 2].axis("off")
            axs[i, 3].imshow(np.abs(pred[i, :, :, 11, 0].T - ACE[i, :, :, 11, 0].T), cmap="hot")
            axs[i, 3].axis("off")

        plt.savefig(f"{self.IMAGE_SAVE_PATH}{epoch}.png", dpi=250)
        plt.close()

    @abstractmethod
    def save_results(self):
        """ Saves json of results """

        json.dump(self.results, open(f"{self.LOG_SAVE_PATH}results.json", 'w'), indent=4)

#-------------------------------------------------------------------------
""" UNet training loop - inherits from BaseTrainingLoop """

class TrainingLoopUNet(BaseTrainingLoop):

    def __init__(self, Model, dataset, config):
        super().__init__(Model, dataset, config)

    def training_loop(self):
        """ Main training loop for UNet """

        self.results = {}
        self.results["train_metric"] = {"global": [], "focal": []}
        self.results["val_metric"] = {"global": [], "focal": []}
        self.results["config"] = self.config
        self.results["epochs"] = []
        self.results["time"] = 0

        start_time = time.time()

        for epoch in range(self.EPOCHS):
            self.results["epochs"].append(epoch + 1)
            self.Model.metric.reset_states()

            for data in self.ds_train:
                self.Model.train_step(data)

            self.results["train_metric"]["global"].append(float(self.Model.metric.result()[0]))
            self.results["train_metric"]["focal"].append(float(self.Model.metric.result()[1]))
            print(f"Train epoch {epoch + 1}, loss [global, focal]: {self.Model.metric.result()}")

            if self.config["EXPT"]["CV_FOLDS"]:
                self.Model.metric.reset_states()

                for data in self.ds_val:
                    self.Model.val_step(data)

                self.results["val_metric"]["global"].append(float(self.Model.metric.result()[0]))
                self.results["val_metric"]["focal"].append(float(self.Model.metric.result()[1]))
                print(f"Val epoch {epoch + 1}, loss [global, focal]: {self.Model.metric.result()}")

            if (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_images(epoch + 1)

        self.results["time"] = (time.time() - start_time) / 3600
        print(f"Time taken: {(time.time() - start_time) / 3600}")
        self.Model.save_weights(f"{self.MODEL_SAVE_PATH}/{self.config['EXPT']['EXPT_NAME']}")
    
    def save_images(self, epoch):
        """ Saves sample of images """

        NCE, ACE, _, _ = next(iter(self.ds_val))
        NCE, ACE = NCE.numpy(), ACE.numpy()
        pred = self.Model(NCE, training=False).numpy()
        super().save_images(NCE, ACE, pred, epoch)
    
    def save_images_ROI(self, epoch):
        """ Saves sample of cropped images """

        NCE, ACE, seg, coords = next(iter(self.ds_val))
        NCEs, ACEs, pred = [], [], []

        for i in range(3):
            rNCE, rACE, _ = self.Model.crop_ROI(NCE, ACE, seg, coords[:, i, :])
            NCEs.append(rNCE[0, ...])
            ACEs.append(rACE[0, ...])
        
        NCEs = tf.stack(NCEs, axis=0)
        ACEs = tf.stack(ACEs, axis=0)

        pred = self.Model(NCEs, training=False).numpy()
        super().save_images(NCEs.numpy(), ACEs.numpy(), pred, epoch)
    
    def save_results(self):
        """ Saves json of results and saves loss curves """

        plt.figure()
        plt.plot(self.results["epochs"], self.results["train_metric"]["global"], 'k-', label="Train global")
        plt.plot(self.results["epochs"], self.results["train_metric"]["focal"], 'k--', label="Train focal")

        if self.config["EXPT"]["CV_FOLDS"]:
            plt.plot(self.results["epochs"], self.results["val_metric"]["global"], 'r-', label="Val global")
            plt.plot(self.results["epochs"], self.results["val_metric"]["focal"], 'r-', label="Val focal")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Losses")
        plt.legend()

        plt.tight_layout()    
        plt.savefig(f"{self.LOG_SAVE_PATH}losses.png")
        plt.close()

        super().save_results()

#-------------------------------------------------------------------------
""" GAN training loop - inherits from BaseTrainingLoop """

class TrainingLoopGAN(BaseTrainingLoop):

    def __init__(self, Model, dataset, config):
        super().__init__(Model, dataset, config)

    # if len(model.generator_metrics.keys()) == 2:
    #     results = {"Discriminator_G": {}, "Discriminator_F": {}}
    # else:
    #     results = {"Discriminator": {}}

    def training_loop(self):
        """ Main training loop for GAN """

        self.results = {}
        self.results["g_metric"] = []
        self.results["d_metric"] = []
        self.results["train_L1"] = {"global": [], "focal": []}
        self.results["val_L1"] = {"global": [], "focal": []}
        self.results["config"] = self.config
        self.results["epochs"] = []
        self.results["time"] = 0

        start_time = time.time()
        best_L1 = 1e6

        for epoch in range(self.EPOCHS):
            self.results["epochs"].append(epoch + 1)

            # for key, value in model.generator_metrics.items():
            self.Model.generator_metric.reset_states()
            self.Model.train_L1_metric.reset_states()

            # for value in model.discriminator_metrics.values():
            self.Model.discriminator_metric.reset_states()

            for data in self.ds_train:
                self.Model.train_step(data)

            # for key, value in model.generator_metrics.items():
            self.results["g_metric"].append(float(self.Model.generator_metric.result()))
            self.results["train_L1"]["global"].append(float(self.Model.train_L1_metric.result()[0]))
            self.results["train_L1"]["focal"].append(float(self.Model.train_L1_metric.result()[1]))
            
            # for key, value in model.discriminator_metrics.items():
            self.results["d_metric"].append(float(self.Model.discriminator_metric.result()))

            # for key in model.discriminator_metrics.keys():
            print(f"Train epoch {epoch + 1}, G: {self.Model.generator_metric.result():.4f} D: {self.Model.discriminator_metric.result():.4f}, L1 [global focal]: {self.Model.train_L1_metric.result()}")

            if self.config["EXPT"]["CV_FOLDS"] != 0:
                self.Model.val_L1_metric.reset_states()

                for data in self.ds_val:
                    self.Model.val_step(data)
                
                self.results["val_L1"]["global"].append(float(self.Model.val_L1_metric.result()[0]))
                self.results["val_L1"]["focal"].append(float(self.Model.val_L1_metric.result()[1]))

                print(f"Val epoch {epoch + 1}, L1 [global focal]: {self.Model.val_L1_metric.result()}")

            if (epoch + 1) % self.SAVE_EVERY == 0:
                if self.config["EXPT"]["CROP"]:
                    self.save_images_ROI(epoch + 1)
                else:
                    self.save_images(epoch + 1)

            if self.Model.val_L1_metric.result()[1] < best_L1 and not self.config["EXPT"]["VERBOSE"]:
                self.Model.save_weights(f"{self.MODEL_SAVE_PATH}{self.config['EXPT']['EXPT_NAME']}")
                best_L1 = self.Model.val_L1_metric.result()[1]
                super().save_results()

        self.results["time"] = (time.time() - start_time) / 3600
        print(f"Time taken: {(time.time() - start_time) / 3600}")
    
    def save_images(self, epoch):
        """ Saves sample of images """

        NCE, ACE, _, _ = next(iter(self.ds_val))
        NCE, ACE = NCE.numpy(), ACE.numpy()
        pred = self.Model.Generator(NCE, training=False).numpy()
        super().save_images(NCE[0:4], ACE[0:4], pred[0:4], epoch)
    
    def save_images_ROI(self, epoch):
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

        pred = self.Model.Generator(NCEs, training=False).numpy()
        super().save_images(NCEs.numpy(), ACEs.numpy(), pred, epoch)

    def save_results(self):
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
        plt.plot(self.results["epochs"], self.results["train_L1"]["global"], 'k--', label="Train global L1")
        plt.plot(self.results["epochs"], self.results["train_L1"]["focal"], 'r--', label="Train focal L1")

        if self.config["EXPT"]["CV_FOLDS"]:
            plt.plot(self.results["epochs"], self.results["val_L1"]["global"], 'k', label="Val global L1")
            plt.plot(self.results["epochs"], self.results["val_L1"]["focal"], 'r', label="Val focal L1")

        plt.xlabel("Epochs")
        plt.ylabel("L1")
        plt.title("Metrics")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.LOG_SAVE_PATH}losses.png")
        plt.close()

        super().save_results()
