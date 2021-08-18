import json
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import tensorflow as tf
import time

from abc import ABC, abstractmethod

np.set_printoptions(suppress=True)


class TrainingGAN:

    def __init__(self, Model: object, dataset: object, val_generator: object, config: dict):
        self.Model = Model
        self.val_generator = val_generator
        self.config = config
        self.EPOCHS = config["expt"]["epochs"]
        self.IMAGE_SAVE_PATH = f"{config['paths']['expt_path']}/images"
        self.MODEL_SAVE_PATH = f"{config['paths']['expt_path']}/models"
        self.LOG_SAVE_PATH = f"{config['paths']['expt_path']}/logs"
        self.SAVE_EVERY = config["expt"]["save_every"]

        self.ds_train, self.ds_val = dataset

        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/train")
        self.test_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/test")

    def train(self, verbose=1):

        """ Main training loop for GAN """

        self.results = {}
        self.results["g_metric"] = []
        self.results["d_metric"] = []
        self.results["train_L1"] = []
        self.results["val_L1"] = []
        self.results["epochs"] = []
        self.results["time"] = 0

        start_time = time.time()
        best_L1 = 1e6

        for epoch in range(self.EPOCHS):
            self.results["epochs"].append(epoch + 1)

            # for key, value in model.generator_metrics.items():
            self.Model.g_metric.reset_states()
            self.Model.train_L1_metric.reset_states()

            # for value in model.discriminator_metrics.values():
            self.Model.d_metric.reset_states()

            # Run training step for each batch in training data
            for data in self.ds_train:
                self.Model.train_step(*data)

            # Log losses
            if self.config["expt"]["log_scalars"]:
                with self.train_writer.as_default():
                    tf.summary.scalar("g_loss", self.Model.g_metric.result(), step=epoch)
                    tf.summary.scalar("d_loss", self.Model.d_metric.result(), step=epoch)

            self.results["g_metric"].append(float(self.Model.g_metric.result()))

            if self.config["expt"]["focal"]:
                self.results["train_L1"].append([float(self.Model.train_L1_metric.result()[0]), float(self.Model.train_L1_metric.result()[1])])

                if self.config["expt"]["log_scalars"]:
                    with self.train_writer.as_default():
                        tf.summary.scalar("train_focal_L1", self.Model.train_L1_metric.result()[0], step=epoch)
                        tf.summary.scalar("train_global_L1", self.Model.train_L1_metric.result()[1], step=epoch)

            else:
                self.results["train_L1"].append(float(self.Model.train_L1_metric.result()))

                if self.config["expt"]["log_scalars"]:
                    with self.train_writer.as_default():
                        tf.summary.scalar("train_L1", self.Model.train_L1_metric.result(), step=epoch)
            
            self.results["d_metric"].append(float(self.Model.d_metric.result()))

            # Log parameter values
            if self.config["expt"]["log_histograms"]:
                with self.train_writer.as_default():
                    for v in self.Model.Generator.trainable_variables:
                        tf.summary.histogram(v.name, v, step=epoch)

            if verbose:
                print(f"Train epoch {epoch + 1}, G: {self.Model.g_metric.result():.4f} D: {self.Model.d_metric.result():.4f}, L1: {self.Model.train_L1_metric.result()}")

            # Validation step if appropriate
            if self.config["data"]["cv_folds"] > 1:
                self.Model.val_L1_metric.reset_states()

                # Run validation step for each batch in validation data
                for data in self.ds_val:
                    self.Model.test_step(*data)

                # Log losses
                if self.config["expt"]["focal"]:
                    self.results["val_L1"].append([float(self.Model.val_L1_metric.result()[0]), float(self.Model.val_L1_metric.result()[1])])

                    if self.config["expt"]["log_scalars"]:
                        with self.test_writer.as_default():
                            tf.summary.scalar("val_focal_L1", self.Model.val_L1_metric.result()[0], step=epoch)
                            tf.summary.scalar("val_global_L1", self.Model.val_L1_metric.result()[1], step=epoch)

                else:
                    self.results["val_L1"].append(float(self.Model.val_L1_metric.result()))

                    if self.config["expt"]["log_scalars"]:
                        with self.test_writer.as_default():
                            tf.summary.scalar("val_L1", self.Model.val_L1_metric.result(), step=epoch)

                if verbose:
                    print(f"Val epoch {epoch + 1}, L1: {self.Model.val_L1_metric.result()}")

            # Save example images
            if (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_images(epoch + 1)

            # if self.Model.val_L1_metric.result() < best_L1 and not self.config["EXPT"]["VERBOSE"]:
            #     self.Model.save_weights(f"{self.MODEL_SAVE_PATH}{self.config['EXPT']['EXPT_NAME']}")
            #     best_L1 = self.Model.val_L1_metric.result()
            #     super().save_results()

        self.results["time"] = (time.time() - start_time) / 3600
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")
    
    def save_images(self, epoch=None, tuning_path=None):

        """ Saves sample of images """

        data = self.val_generator.example_images()

        if len(data) == 2:
            source, target = data
        else:
            source, target, seg = data

        # Spatial transformer if necessary
        if self.Model.STN is not None and len(data) == 2:
            target, _ = self.Model.STN(source=source, target=target, print_matrix=False)
        elif self.Model.STN is not None and len(data) == 3:
            target, _ = self.Model.STN(source=source, target=target, seg=seg, print_matrix=False)
        else:
            pass

        pred = self.Model.Generator(source, training=False).numpy()

        source = self.val_generator.un_normalise(source)
        target = self.val_generator.un_normalise(target)
        pred = self.val_generator.un_normalise(pred)

        fig, axs = plt.subplots(target.shape[0], 5)

        for i in range(target.shape[0]):
            axs[i, 0].imshow(source[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 0].axis("off")
            axs[i, 1].imshow(target[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 1].axis("off")
            axs[i, 3].imshow(np.abs(target[i, :, :, 11, 0] - source[i, :, :, 11, 0]), cmap="hot")
            axs[i, 3].axis("off")
            axs[i, 2].imshow(pred[i, :, :, 11, 0], cmap="gray", vmin=-150, vmax=250)
            axs[i, 2].axis("off")
            axs[i, 4].imshow(np.abs(target[i, :, :, 11, 0] - pred[i, :, :, 11, 0]), cmap="hot")
            axs[i, 4].axis("off")

        plt.tight_layout()

        if tuning_path:
            plt.savefig(f"{tuning_path}.png", dpi=250)
        else:
            plt.savefig(f"{self.IMAGE_SAVE_PATH}/{epoch}.png", dpi=250)

        plt.close()

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

        if self.config["expt"]["focal"]:
            plt.plot(self.results["epochs"], np.array(self.results["train_L1"])[:, 0], 'k-', label="Train global L1")
            plt.plot(self.results["epochs"], np.array(self.results["train_L1"])[:, 1], 'k--', label="Train global L1")

            if self.config["data"]["cv_folds"]:
                plt.plot(self.results["epochs"], np.array(self.results["val_L1"])[:, 0], 'r-', label="Val global L1")
                plt.plot(self.results["epochs"], np.array(self.results["val_L1"])[:, 1], 'r--', label="Val global L1")
        
        else:
            plt.plot(self.results["epochs"], self.results["train_L1"], 'k-', label="Train L1")

            if self.config["data"]["cv_folds"]:
                plt.plot(self.results["epochs"], self.results["val_L1"], 'r-', label="Val L1")

        plt.xlabel("Epochs")
        plt.ylabel("L1")
        plt.title("Metrics")
        plt.legend()

        if tuning_path:
            plt.savefig(tuning_path)
        else:
            plt.savefig(f"{self.LOG_SAVE_PATH}/losses.png")
            json.dump(self.results, open(f"{self.LOG_SAVE_PATH}/results.json", 'w'), indent=4)
        
        plt.close()
