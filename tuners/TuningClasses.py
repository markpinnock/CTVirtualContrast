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
    
    def __init__(self, CONFIG, tuning_config, TrainingLoop):
        self.all_run_results = {}
        self.current_run_results = {}
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
        
        """ ========================================================================== """
        """ WARNING: pass ONLY dict initialised as empty to avoid shallow copy problem """
        """ ========================================================================== """
        
        results = {key: self.current_run_results[key] for key in ["train", "validation"]}
        config = self.current_run_results["config"]
        self.all_run_results[run_name] = {"results": results, "config": config}

        print(f"Final train loss {results}")
        json.dump(self.all_run_results, open(f"{self.RESULTS_PATH}tuning_results.json", 'w'), indent=4)
        
        # Saves loss curves from training loop
        self.Train.save_results(f"{self.RESULTS_PATH}losses_{run_name}.png")

#-------------------------------------------------------------------------
""" Grid search algorithm - inherits from BaseTuner """

class GridSearch(BaseTuner):

    def __init__(self, EXPT_NAME, CONFIG, tuning_config, TrainingLoop):
        super().__init__(CONFIG, tuning_config, TrainingLoop)
        self.CONFIG["EXPT"]["EXPT_NAME"] = "GridSearch"
        self.RESULTS_PATH = f"{CONFIG['EXPT']['SAVE_PATH']}tuning/GridSearch/{CONFIG['EXPT']['MODEL']}/{EXPT_NAME}/"
        if not os.path.exists(self.RESULTS_PATH): os.makedirs(self.RESULTS_PATH)
        self.tuning_config = tuning_config["GRID"]
    
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
                self.Train.save_images_ROI(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}")
            else:
                self.Train.save_images(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}")
    
    def save_results(self, run_name: str):
        """ Save current results
            - run_name: hyper-parameter combination """

        """ ============================================================================== """
        """ WARNING: save results and config ONLY empty dict to avoid shallow copy problem """
        """ ============================================================================== """

        self.current_run_results = {}
        results = self.Train.results
        config = self.Train.config
        self.current_run_results["train"] = [results["train_metric"]["global"][-1], results["train_metric"]["focal"][-1], results["train_metric"]["weights"][-1]]
        self.current_run_results["validation"] = [results["val_metric"]["global"][-1], results["val_metric"]["focal"][-1], results["train_metric"]["weights"][-1]]
        self.current_run_results["config"] = config
        super().save_results(run_name)

#-------------------------------------------------------------------------
""" Random search algorithm - inherits from BaseTuner """

class RandomSearch(BaseTuner):

    def __init__(self, EXPT_NAME, CONFIG, tuning_config, TrainingLoop):
        super().__init__(CONFIG, tuning_config, TrainingLoop)
        self.CONFIG["EXPT"]["EXPT_NAME"] = "RandomSearch"
        self.RESULTS_PATH = f"{CONFIG['EXPT']['SAVE_PATH']}tuning/RandomSearch/{CONFIG['EXPT']['MODEL']}/{EXPT_NAME}/"
        if not os.path.exists(self.RESULTS_PATH): os.makedirs(self.RESULTS_PATH)
        self.tuning_config = tuning_config["RANDOM"]
    
    def tuning_loop(self, runs):
        """ Runs training process for hyper-parameters """

        np.random.seed()

        for i in range(runs):
            run_name = ""
            new_config = {key: val for key, val in self.CONFIG.items()}
            ROI = None

            for key, val in self.tuning_config.items():
                if key in ["ETA", "D_ETA", "G_ETA", "LAMBDA"]:
                    new_val = float(np.power(10.0, np.random.uniform(val[0], val[1])))
                    run_name += f"{key}_{new_val:.6f}_"
            
                elif key in ["MU"]:
                    new_val = float(np.random.uniform(val[0], val[1]))
                    run_name += f"{key}_{new_val:.2f}_"

                elif key in ["NF", "NDF_G", "NGF", "D_IN_CH"]:
                    new_val = int(np.random.choice(val))
                    run_name += f"{key}_{new_val}_"
                
                elif key in ["ROI"]:
                    ROI = int(np.random.choice(val))
                    new_config["EXPT"]["IMG_DIMS"] = [ROI, ROI, 12]
                    run_name += f"{key}_{ROI}_"
                    continue

                elif key in ["D_LAYERS_G"]:
                    assert ROI, "ROI needs to come before D_LAYERS_G"
                    new_val = int(np.random.randint(1, np.log2(ROI / 4)))
                    run_name += f"{key}_{new_val}_"

                elif key in ["G_LAYERS"]:
                    assert ROI, "ROI needs to come before G_LAYERS"
                    new_val = int(np.random.randint(2, np.log2(ROI)))

                    # TODO: need a better solution to max_z_downsample == num_layers bug
                    if new_val == 3:
                        if np.random.rand() > 0.5:
                            new_val = 4
                        else:
                            new_val = 2
                    
                    run_name += f"{key}_{new_val}_"

                elif key in ["GAMMA"]:
                    continue

                else:
                    raise ValueError("Key not recognised")

                new_config["HYPERPARAMS"][key] = new_val
            
            run_name = run_name.strip('_')
            
            # Select ROI or base model
            if ROI == 512 // new_config["EXPT"]["DOWN_SAMP"]:
                new_config["EXPT"]["CROP"] = 0

                if self.CONFIG["EXPT"]["MODEL"] == "GAN":
                    Model = GAN(new_config)
                elif self.CONFIG["EXPT"]["MODEL"] == "UNet":
                    Model = UNet(new_config)
                else:
                    raise ValueError("Model not recognised")

            else:
                new_config["EXPT"]["CROP"] = 1

                if self.CONFIG["EXPT"]["MODEL"] == "GAN":
                    Model = CropGAN(new_config)
                elif self.CONFIG["EXPT"]["MODEL"] == "UNet":
                    Model = CropUNet(new_config)
                else:
                    raise ValueError("Model not recognised")

            self.Train = self.TrainingLoop(Model=Model, dataset=(self.train_ds, self.val_ds), config=new_config)

            print("=================================================")
            print(f"{run_name} ({i + 1} of {runs})")

            self.Train.training_loop(verbose=1)
            self.save_results(run_name)

            # Save sample images
            if self.Train.config["EXPT"]["CROP"]:
                self.Train.save_images_ROI(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}")
            else:
                self.Train.save_images(epoch=None, tuning_path=f"{self.RESULTS_PATH}images_{run_name}")
    
    def save_results(self, run_name: str):
        """ Save current results
            - run_name: hyper-parameter combination """

        """ ============================================================================== """
        """ WARNING: save results and config ONLY empty dict to avoid shallow copy problem """
        """ ============================================================================== """

        self.current_run_results = {}
        results = self.Train.results
        config = self.Train.config

        # TODO: standardise metric terminology across models
        if self.CONFIG["EXPT"]["MODEL"] == "GAN":
            self.current_run_results["train"] = [results["train_L1"]["global"][-1], results["train_L1"]["focal"][-1], results["train_L1"]["weights"][-1]]
            self.current_run_results["validation"] = [results["val_L1"]["global"][-1], results["val_L1"]["focal"][-1], results["val_L1"]["weights"][-1]]
        elif self.CONFIG["EXPT"]["MODEL"] == "UNet":
            self.current_run_results["train"] = [results["train_metric"]["global"][-1], results["train_metric"]["focal"][-1], results["train_metric"]["weights"][-1]]
            self.current_run_results["validation"] = [results["val_metric"]["global"][-1], results["val_metric"]["focal"][-1], results["val_metric"]["weights"][-1]]
        else:
            raise ValueError("Model not recognised")
        
        self.current_run_results["config"] = config
        super().save_results(run_name)
