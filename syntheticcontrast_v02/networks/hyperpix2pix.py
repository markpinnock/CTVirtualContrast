import numpy as np
import tensorflow as tf

from .pix2pix import Pix2Pix
from .components.hypergenerator import HyperGenerator


""" Wrapper for Pix2pix GAN with HyperNetwork """

class HyperPix2Pix(Pix2Pix):

    """ GAN class using HyperNetwork for generator """

    def __init__(self, config: dict, name: str = "HyperGAN"):
        super().__init__(config, name=name)

    def generator_init(self, config):
        # Check generator output dims match input
        G_input_size = [1] + self.img_dims + [1]
        G_output_size = [1] + self.img_dims + [1]

        self.Generator = HyperGenerator(self.initialiser, config, name="generator")

        if self.input_times:
            assert self.Generator.build_model(tf.zeros(G_input_size), tf.zeros(1)) == G_output_size, f"{self.Generator.build_model(tf.zeros(G_input_size), tf.zeros(1))} vs {G_input_size}"
        else:
            assert self.Generator.build_model(tf.zeros(G_input_size)) == G_output_size, f"{self.Generator.build_model(tf.zeros(G_input_size))} vs {G_input_size}"

    def summary(self):
        source = tf.keras.Input(shape=self.img_dims + [1])

        if self.input_times:
            outputs = self.Generator.call(source, tf.zeros(1))
        else:
            outputs = self.Generator.call(source)

        print("===========================================================")
        print(f"Generator: {np.sum([np.prod(v.shape) for v in self.Generator.trainable_variables])}")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
        source = tf.keras.Input(shape=self.img_dims + [2])

        if self.input_times:
            outputs = self.Discriminator.call(source, tf.zeros(1))
        else:
            outputs = self.Discriminator.call(source)    

        print("===========================================================")
        print(f"Discriminator: {np.sum([np.prod(v.shape) for v in self.Discriminator.trainable_variables])}")
        print("===========================================================")
        tf.keras.Model(inputs=source, outputs=outputs).summary()
