from .training_unet import TrainingUNet
from .training_pix2pix import TrainingPix2Pix
from .training_hyperpix2pix import TrainingHyperPix2Pix
from. training_cyclegan import TrainingCycleGAN


def get_training_loop(Model: object,
                      dataset: object,
                      train_generator: object,
                      val_generator: object,
                      config: dict):

    training_loops = {"UNet": TrainingUNet,
                      "Pix2Pix": TrainingPix2Pix,
                      "HyperPix2Pix": TrainingHyperPix2Pix,
                      "CycleGAN": TrainingCycleGAN}
    model_type = config["expt"]["model"]

    return training_loops[model_type](Model,
                                      dataset,
                                      train_generator,
                                      val_generator,
                                      config)
