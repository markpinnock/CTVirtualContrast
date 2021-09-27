import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

np.set_printoptions(suppress=True)


#-------------------------------------------------------------------------

class TrainingPix2Pix:

    def __init__(self, Model: object, dataset: object, train_generator: object, val_generator: object, config: dict):
        self.Model = Model
        self.config = config
        self.EPOCHS = config["expt"]["epochs"]
        self.IMAGE_SAVE_PATH = f"{config['paths']['expt_path']}/images"
        self.MODEL_SAVE_PATH = f"{config['paths']['expt_path']}/models"
        self.LOG_SAVE_PATH = f"{config['paths']['expt_path']}/logs"
        self.SAVE_EVERY = config["expt"]["save_every"]

        if not os.path.exists(f"{self.IMAGE_SAVE_PATH}/train"):
            os.makedirs(f"{self.IMAGE_SAVE_PATH}/train")

        if not os.path.exists(f"{self.IMAGE_SAVE_PATH}/validation"):
            os.makedirs(f"{self.IMAGE_SAVE_PATH}/validation")

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.ds_train, self.ds_val = dataset

        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/train")
        self.test_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/test")

    def train(self, verbose=1):

        """ Main training loop for Pix2Pix """

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
            self.Model.reset_train_metrics()

            # Run training step for each batch in training data
            for data in self.ds_train:
                self.Model.train_step(**data)

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
                        tf.summary.scalar("focal_L1", self.Model.train_L1_metric.result()[0], step=epoch)
                        tf.summary.scalar("global_L1", self.Model.train_L1_metric.result()[1], step=epoch)

            else:
                self.results["train_L1"].append(float(self.Model.train_L1_metric.result()))

                if self.config["expt"]["log_scalars"]:
                    with self.train_writer.as_default():
                        tf.summary.scalar("L1", self.Model.train_L1_metric.result(), step=epoch)
            
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
                    self.Model.test_step(**data)

                # Log losses
                if self.config["expt"]["focal"]:
                    self.results["val_L1"].append([float(self.Model.val_L1_metric.result()[0]), float(self.Model.val_L1_metric.result()[1])])

                    if self.config["expt"]["log_scalars"]:
                        with self.test_writer.as_default():
                            tf.summary.scalar("focal_L1", self.Model.val_L1_metric.result()[0], step=epoch)
                            tf.summary.scalar("global_L1", self.Model.val_L1_metric.result()[1], step=epoch)

                else:
                    self.results["val_L1"].append(float(self.Model.val_L1_metric.result()))

                    if self.config["expt"]["log_scalars"]:
                        with self.test_writer.as_default():
                            tf.summary.scalar("L1", self.Model.val_L1_metric.result(), step=epoch)

                if verbose:
                    print(f"Val epoch {epoch + 1}, L1: {self.Model.val_L1_metric.result()}")

            # Save example images
            if (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_images(epoch + 1, phase="train")
                self.save_images(epoch + 1, phase="validation")

            # if self.Model.val_L1_metric.result() < best_L1 and not self.config["EXPT"]["VERBOSE"]:
            #     self.Model.save_weights(f"{self.MODEL_SAVE_PATH}{self.config['EXPT']['EXPT_NAME']}")
            #     best_L1 = self.Model.val_L1_metric.result()
            #     super().save_results()

        self.results["time"] = (time.time() - start_time) / 3600
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")
    
    def save_images(self, epoch, phase="validation", tuning_path=None):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images()
        pred = self.Model.Generator(data["real_source"], data["source_times"], data["target_times"]).numpy()

        source = data_generator.un_normalise(data["real_source"])
        target = data_generator.un_normalise(data["real_target"])
        pred = data_generator.un_normalise(pred)

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
            plt.savefig(f"{self.IMAGE_SAVE_PATH}/{phase}/{epoch}.png", dpi=250)

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
            plt.plot(self.results["epochs"], np.array(self.results["train_L1"])[:, 0], 'k-', label="Train focal L1")
            plt.plot(self.results["epochs"], np.array(self.results["train_L1"])[:, 1], 'k--', label="Train global L1")

            if self.config["data"]["cv_folds"]:
                plt.plot(self.results["epochs"], np.array(self.results["val_L1"])[:, 0], 'r-', label="Val focal L1")
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


#-------------------------------------------------------------------------

class TrainingCycleGAN:

    def __init__(self, Model: object, dataset: object, train_generator: object, val_generator: object, config: dict):
        self.Model = Model
        self.config = config
        self.EPOCHS = config["expt"]["epochs"]
        self.IMAGE_SAVE_PATH = f"{config['paths']['expt_path']}/images"
        self.MODEL_SAVE_PATH = f"{config['paths']['expt_path']}/models"
        self.LOG_SAVE_PATH = f"{config['paths']['expt_path']}/logs"
        self.SAVE_EVERY = config["expt"]["save_every"]

        if not os.path.exists(f"{self.IMAGE_SAVE_PATH}/train"):
            os.makedirs(f"{self.IMAGE_SAVE_PATH}/train")

        if not os.path.exists(f"{self.IMAGE_SAVE_PATH}/validation"):
            os.makedirs(f"{self.IMAGE_SAVE_PATH}/validation")

        self.train_generator = train_generator
        self.val_generator = val_generator
        self.ds_train, self.ds_val = dataset

        log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/train")
        self.test_writer = tf.summary.create_file_writer(f"{self.LOG_SAVE_PATH}/{log_time}/test")

    def train(self, verbose=1):

        """ Main training loop for CycleGAN """

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
            self.Model.reset_train_metrics()

            # Run training step for each batch in training data
            for data in self.ds_train:
                self.Model.train_step(*data)

            # Log losses
            if self.config["expt"]["log_scalars"]:
                with self.train_writer.as_default():
                    tf.summary.scalar("g_loss", self.Model.g_forward_metric.result(), step=epoch)
                    tf.summary.scalar("d_loss", self.Model.d_forward_metric.result(), step=epoch)

            self.results["g_metric"].append(float(self.Model.g_forward_metric.result()))

            if self.config["expt"]["focal"]:
                self.results["train_L1"].append([float(self.Model.train_L1_metric.result()[0]), float(self.Model.train_L1_metric.result()[1])])

                if self.config["expt"]["log_scalars"]:
                    with self.train_writer.as_default():
                        tf.summary.scalar("focal_L1", self.Model.train_L1_metric.result()[0], step=epoch)
                        tf.summary.scalar("global_L1", self.Model.train_L1_metric.result()[1], step=epoch)

            else:
                self.results["train_L1"].append(float(self.Model.train_L1_metric.result()))

                if self.config["expt"]["log_scalars"]:
                    with self.train_writer.as_default():
                        tf.summary.scalar("L1", self.Model.train_L1_metric.result(), step=epoch)
            
            self.results["d_metric"].append(float(self.Model.d_forward_metric.result()))

            # Log parameter values
            if self.config["expt"]["log_histograms"]:
                with self.train_writer.as_default():
                    for v in self.Model.G_forward.trainable_variables:
                        tf.summary.histogram(v.name, v, step=epoch)

            if verbose:
                print(f"Train epoch {epoch + 1}, G: {self.Model.g_forward_metric.result():.4f} D: {self.Model.d_forward_metric.result():.4f}, L1: {self.Model.train_L1_metric.result()}")

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
                            tf.summary.scalar("focal_L1", self.Model.val_L1_metric.result()[0], step=epoch)
                            tf.summary.scalar("global_L1", self.Model.val_L1_metric.result()[1], step=epoch)

                else:
                    self.results["val_L1"].append(float(self.Model.val_L1_metric.result()))

                    if self.config["expt"]["log_scalars"]:
                        with self.test_writer.as_default():
                            tf.summary.scalar("L1", self.Model.val_L1_metric.result(), step=epoch)

                if verbose:
                    print(f"Val epoch {epoch + 1}, L1: {self.Model.val_L1_metric.result()}")

            # Save example images
            if (epoch + 1) % self.SAVE_EVERY == 0:
                self.save_images(epoch + 1, phase="train")
                self.save_images(epoch + 1, phase="validation")

            # if self.Model.val_L1_metric.result() < best_L1 and not self.config["EXPT"]["VERBOSE"]:
            #     self.Model.save_weights(f"{self.MODEL_SAVE_PATH}{self.config['EXPT']['EXPT_NAME']}")
            #     best_L1 = self.Model.val_L1_metric.result()
            #     super().save_results()

        self.results["time"] = (time.time() - start_time) / 3600
        
        if verbose:
            print(f"Time taken: {(time.time() - start_time) / 3600}")
    
    def save_images(self, epoch, phase="validation", tuning_path=None):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images()

        if len(data) == 2:
            source, target = data
        else:
            source, target, seg = data

        pred = self.Model.G_forward(source, training=False).numpy()

        source = data_generator.un_normalise(source)
        target = data_generator.un_normalise(target)
        pred = data_generator.un_normalise(pred)

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
            plt.savefig(f"{self.IMAGE_SAVE_PATH}/{phase}/{epoch}.png", dpi=250)

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
            plt.plot(self.results["epochs"], np.array(self.results["train_L1"])[:, 0], 'k-', label="Train focal L1")
            plt.plot(self.results["epochs"], np.array(self.results["train_L1"])[:, 1], 'k--', label="Train global L1")

            if self.config["data"]["cv_folds"]:
                plt.plot(self.results["epochs"], np.array(self.results["val_L1"])[:, 0], 'r-', label="Val focal L1")
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
