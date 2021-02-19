import json
import numpy as np
import os
import tensorflow as tf
from abc import ABC, abstractmethod

from networks.GANWrapper import GAN, CropGAN
from networks.UNet import UNet, CropUNet
from utils.DataLoader import OneToOneLoader


#-------------------------------------------------------------------------
""" Base class for tuning algorithms """

class BaseTuner(ABC):
    
    def __init__(self, CONFIG, TrainingLoop):
        self.tuning_results = {}
        self.curr_results = {}
        self.CONFIG = CONFIG
        self.CONFIG["EXPT"]["SAVE_EVERY"] = 100
        self.TrainingLoop = TrainingLoop
        self.Train = None

        # Initialise datasets
        TrainGenerator = OneToOneLoader(config=CONFIG["EXPT"], dataset_type="training", fold=5)
        ValGenerator = OneToOneLoader(config=CONFIG["EXPT"], dataset_type="validation", fold=5)

        # Batch size (separate batches for generator and critic runs)
        if CONFIG["EXPT"]["MODEL"] == "GAN":
            MB_SIZE = CONFIG["HYPERPARAMS"]["MB_SIZE"] + CONFIG["HYPERPARAMS"]["MB_SIZE"] * CONFIG["HYPERPARAMS"]["N_CRITIC"]
        else:
            MB_SIZE = CONFIG["HYPERPARAMS"]["MB_SIZE"]

        # Create data generators
        self.train_ds = tf.data.Dataset.from_generator(
            generator=TrainGenerator.data_generator,
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
            ).batch(MB_SIZE)

        self.val_ds = tf.data.Dataset.from_generator(
            generator=ValGenerator.data_generator,
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32)
            ).batch(MB_SIZE)

    @abstractmethod
    def tuning_loop(self):
        raise NotImplementedError

    @abstractmethod
    def save_results(self, run_name: str):
        """ Save current results and configuration
            - run_name: hyper-parameter combination """
        
        self.tuning_results[run_name] = {"results": self.curr_results, "config": self.Train.config}
        print(f"Final train loss {self.curr_results}")
        json.dump(self.tuning_results, open(f"{self.RESULTS_PATH}tuning_results.json", 'w'), indent=4)
        
        # Saves loss curves from training loop
        self.Train.save_results(f"{self.RESULTS_PATH}losses_{run_name}.png")

#-------------------------------------------------------------------------
""" Grid search algorithm - inherits from BaseTuner """

class GridSearch(BaseTuner):

    def __init__(self, EXPT_NAME, CONFIG, TrainingLoop):
        super().__init__(CONFIG, TrainingLoop)
        self.CONFIG["EXPT"]["EXPT_NAME"] = "GridSearch"
        self.RESULTS_PATH = f"{CONFIG['EXPT']['SAVE_PATH']}tuning/GridSearch/{CONFIG['EXPT']['MODEL']}/{EXPT_NAME}/"
        if not os.path.exists(self.RESULTS_PATH): os.makedirs(self.RESULTS_PATH)
    
    def tuning_loop(self):
        """ Runs training process for hyper-parameters """

        etas = [float(np.power(10.0, -i)) for i in range(2, 5)]
        ROIs = [int(512 / (np.power(2, x) * self.CONFIG["EXPT"]["DOWN_SAMP"])) for x in range(4)]
        ROI_grid, eta_grid = np.meshgrid(etas, ROIs)
        hyper_params = np.vstack([eta_grid.ravel(), ROI_grid.ravel()])
        runs = hyper_params.shape[1]

        for i in range(runs):

            self.CONFIG["HYPERPARAMS"]["ETA"] = hyper_params[1, i]
            self.CONFIG["EXPT"]["IMG_DIMS"] = [int(hyper_params[0, i]), int(hyper_params[0, i]), 12]

            # Select ROI or base model
            if hyper_params[0, i] == 512 // self.CONFIG["EXPT"]["DOWN_SAMP"]:
                self.CONFIG["EXPT"]["CROP"] = 0
                Model = UNet(self.CONFIG)
            else:
                self.CONFIG["EXPT"]["CROP"] = 1
                Model = CropUNet(self.CONFIG)

            run_name = f"ETA_{self.CONFIG['HYPERPARAMS']['ETA']:.5f}"\
                    f"_ROI_{int(hyper_params[0, i])}"
            
            self.Train = self.TrainingLoop(Model=Model, dataset=(self.train_ds, self.val_ds), config=self.CONFIG)

            print("=================================================")
            print(f"{run_name} ({i + 1} of {runs})")

            self.Train.training_loop(verbose=0)
            self.save_results(run_name)

            # Save sample images
            if self.Train.config["EXPT"]["CROP"]:
                self.Train.save_images_ROI(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}.png")
            else:
                self.Train.save_images(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}.png")
    
    def save_results(self, run_name: str):
        """ Save current results
            - run_name: hyper-parameter combination """

        results = self.Train.results
        self.curr_results["train"] = [results["train_metric"]["global"][-1], results["train_metric"]["focal"][-1]]
        self.curr_results["validation"] = [results["val_metric"]["global"][-1], results["val_metric"]["focal"][-1]]
        super().save_results(run_name)

#-------------------------------------------------------------------------
""" Random search algorithm - inherits from BaseTuner """

class RandomSearch(BaseTuner):

    def __init__(self, EXPT_NAME, CONFIG, TrainingLoop):
        super().__init__(CONFIG, TrainingLoop)
        self.CONFIG["EXPT"]["EXPT_NAME"] = "RandomSearch"
        self.RESULTS_PATH = f"{CONFIG['EXPT']['SAVE_PATH']}tuning/RandomSearch/{CONFIG['EXPT']['MODEL']}/{EXPT_NAME}/"
        if not os.path.exists(self.RESULTS_PATH): os.makedirs(self.RESULTS_PATH)
    
    def tuning_loop(self, runs):
        """ Runs training process for hyper-parameters """

        np.random.seed()

        for i in range(runs):

            self.CONFIG["HYPERPARAMS"]["D_ETA"] = float(np.power(10.0, -np.random.uniform(2, 5)))
            self.CONFIG["HYPERPARAMS"]["G_ETA"] = float(np.power(10.0, -np.random.uniform(2, 5)))
            self.CONFIG["HYPERPARAMS"]["LAMBDA"] = float(np.power(10.0, np.random.uniform(0, 3)))
            
            if self.CONFIG["HYPERPARAMS"]["LAMBDA"]:
                self.CONFIG["HYPERPARAMS"]["MU"] = float(np.random.uniform(0, 1))
            else:
                self.CONFIG["HYPERPARAMS"]["MU"] = 0.0

            self.CONFIG["HYPERPARAMS"]["NDF_G"] = int(np.random.choice([16, 32, 64]))
            self.CONFIG["HYPERPARAMS"]["NGF"] = int(np.random.choice([16, 32, 64]))
            self.CONFIG["HYPERPARAMS"]["D_IN_CH"] = int(np.random.choice([2, 3]))
            ROI = int(np.random.choice([512, 256, 128, 64]) / self.CONFIG["EXPT"]["DOWN_SAMP"])
            self.CONFIG["EXPT"]["IMG_DIMS"] = [ROI, ROI, 12]
            self.CONFIG["HYPERPARAMS"]["D_LAYERS_G"] = int(np.random.randint(1, np.log2(ROI / 4)))
            self.CONFIG["HYPERPARAMS"]["G_LAYERS"] = int(np.random.randint(2, np.log2(ROI)))
            
            # TODO: need a better solution to max_z_downsample == num_layers bug
            if self.CONFIG["HYPERPARAMS"]["G_LAYERS"] == 3:
                if np.random.rand() > 0.5:
                    self.CONFIG["HYPERPARAMS"]["G_LAYERS"] = 4
                else:
                    self.CONFIG["HYPERPARAMS"]["G_LAYERS"] = 2

            # Select ROI or base model
            if ROI == 512 // self.CONFIG["EXPT"]["DOWN_SAMP"]:
                self.CONFIG["EXPT"]["CROP"] = 0
                Model = GAN(self.CONFIG)
            else:
                self.CONFIG["EXPT"]["CROP"] = 1
                Model = CropGAN(self.CONFIG)

            run_name = f"DETA_{self.CONFIG['HYPERPARAMS']['D_ETA']:.6f}_"\
                       f"GETA_{self.CONFIG['HYPERPARAMS']['G_ETA']:.6f}_"\
                       f"LAMBDA_{self.CONFIG['HYPERPARAMS']['LAMBDA']:.2f}_"\
                       f"MU_{self.CONFIG['HYPERPARAMS']['MU']:.2f}_"\
                       f"DNF_{self.CONFIG['HYPERPARAMS']['NDF_G']}_"\
                       f"GNF_{self.CONFIG['HYPERPARAMS']['NGF']}_"\
                       f"DLA_{self.CONFIG['HYPERPARAMS']['D_LAYERS_G']}_"\
                       f"GLA_{self.CONFIG['HYPERPARAMS']['G_LAYERS']}_"\
                       f"INCH_{self.CONFIG['HYPERPARAMS']['D_IN_CH']}_"\
                       f"ROI_{ROI}"

            self.Train = self.TrainingLoop(Model=Model, dataset=(self.train_ds, self.val_ds), config=self.CONFIG)

            print("=================================================")
            print(f"{run_name} ({i + 1} of {runs})")

            self.Train.training_loop(verbose=0)
            self.save_results(run_name)

            # Save sample images
            if self.Train.config["EXPT"]["CROP"]:
                self.Train.save_images_ROI(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}.png")
            else:
                self.Train.save_images(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}.png")
    
    def save_results(self, run_name: str):
        """ Save current results
            - run_name: hyper-parameter combination """

        results = self.Train.results
        self.curr_results["train"] = [results["train_L1"]["global"][-1], results["train_L1"]["focal"][-1]]
        self.curr_results["validation"] = [results["val_L1"]["global"][-1], results["val_L1"]["focal"][-1]]
        super().save_results(run_name)
