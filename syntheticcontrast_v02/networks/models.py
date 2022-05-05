import abc
import tensorflow as tf

from .cyclegan import CycleGAN
from .hyperpix2pix import HyperPix2Pix
from .pix2pix import Pix2Pix
from .unet import UNet


#-------------------------------------------------------------------------

class Model(abc.ABC):
    def __init__(self, config: dict):
        self.model_dict = {"UNet": UNet,
                           "Pix2Pix": Pix2Pix,
                           "HyperPix2Pix": HyperPix2Pix,
                           "CycleGAN": CycleGAN}
        self.config = config
        self._model = None

    def build_model(self):
        model_type = self.config["expt"]["model"]
        if model_type not in self.model_dict.keys():
            raise ValueError(f"Invalid model type: {model_type}")

        self._model = self.model_dict[model_type](self.config)

    @abc.abstractmethod
    def compile_model(self):
        pass

    @abc.abstractmethod
    def load_weights(self):
        pass

    @property
    def model(self):
        return self._model

#-------------------------------------------------------------------------

class UNetModel(Model):

    def __init__(self, config: dict):
        super().__init__(config)

    def compile_model(self):
        self.model.compile(
            optimiser=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["opt"],
                name="opt"))

    def load_weights(self):
        self._model.UNet.load_weights(
        f"{self.config['paths']['expt_path']}/models/model.ckpt")


#-------------------------------------------------------------------------

class Pix2PixModel(Model):

    def __init__(self, config: dict):
        super().__init__(config)

    def compile_model(self):
        self.model.compile(
            g_optimiser=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["g_opt"],
                name="g_opt"),
            d_optimiser=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["d_opt"],
                name="d_opt"))

    def load_weights(self):
        self._model.Generator.load_weights(
        f"{self.config['paths']['expt_path']}/models/generator.ckpt")


#-------------------------------------------------------------------------

class CycleGANModel(Model):

    def __init__(self, config: dict):
        super().__init__(config)

    def compile_model(self):
        self.model.compile(
            g_forward_opt=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["g_opt"],
                name="g_foward_opt"),
            g_backward_opt=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["g_opt"],
                name="g_backward_opt"),
            d_forward_opt=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["d_opt"],
                name="d_forward_opt"),
            d_backward_opt=tf.keras.optimizers.Adam(
                *self.config["hyperparameters"]["d_opt"],
                name="d_backward_opt"))

    def load_weights(self):
        self._model.Generator.load_weights(
        f"{self.config['paths']['expt_path']}/models/generator.ckpt")


#-------------------------------------------------------------------------

def get_model(config: dict, purpose: str = "training"):
    model_factory_dict = {"UNet": UNetModel,
                          "Pix2Pix": Pix2PixModel,
                          "HyperPix2Pix": Pix2PixModel,
                          "CycleGAN": CycleGANModel}
    model_type = config["expt"]["model"]
    model_factory = model_factory_dict[model_type](config)
    model_factory.build_model()

    if purpose == "training":
        model_factory.compile_model()
        return model_factory.model
    elif purpose == "inference":
        model_factory.load_weights()
        return model_factory.model
    else:
        raise ValueError("Purpose must be 'training' or 'inference'")
