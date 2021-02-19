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

    def __init__(self, CONFIG, TrainingLoop):
        super().__init__(CONFIG, TrainingLoop)
        self.CONFIG["EXPT"]["EXPT_NAME"] = "GridSearch"
        self.RESULTS_PATH = f"{CONFIG['EXPT']['SAVE_PATH']}tuning/GridSearch/{CONFIG['EXPT']['MODEL']}/"
        if not os.path.exists(self.RESULTS_PATH): os.makedirs(self.RESULTS_PATH)
    
    def tuning_loop(self):
        """ Runs training process for hyper-parameters """

        etas = [float(np.power(10.0, -i)) for i in range(1, 6)]
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

    def __init__(self, CONFIG, TrainingLoop):
        super().__init__(CONFIG, TrainingLoop)
        self.CONFIG["EXPT"]["EXPT_NAME"] = "Random"
    
    def tuning_loop(self):
        """ Runs training process for hyper-parameters """

        np.random.seed(5)

        for i in range(RUNS):

            self.CONFIG["HYPERPARAMS"]["ETA"] = float(np.power(10.0, np.random.uniform(-1, -6)))
            num_layers = self.CONFIG["HYPERPARAMS"]["LAYERS"]
            self.CONFIG["HYPERPARAMS"]["NF"] = [int(np.power(2, np.random.randint(3, 10))) for _ in range(num_layers)]

            run_name = f"ETA_{self.CONFIG['EXPT']['ETA']:.6f}"\
                    f"_NF_{str(self.CONFIG['MODEL_CONFIG']['UNITS'])}"
            
            # Initialise model and training loop with current hyper-parameters
            Model = self.Model(self.CONFIG)
            self.Train = TrainingLoop(Model=Model, dataset=(self.train_ds, self.val_ds), config=self.CONFIG)

            print("=================================================")
            print(f"{run_name} ({i + 1} of {runs})")

            self.Train.training_loop(verbose=0)
            self.save_results(run_name)
    
    def save_results(self, run_name: str):
        """ Save current results
            - run_name: hyper-parameter combination """

        raise NotImplementedError
        super().save_results(run_name)
