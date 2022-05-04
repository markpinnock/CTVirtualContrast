import numpy as np
import tensorflow as tf

from .layer.layers import GANDownBlock


""" PatchGAN discriminator for GAN-based methods """

class Discriminator(tf.keras.Model):

    """ Input:
        - initialiser: e.g. keras.initializers.RandomNormal
        - config: configuration json
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super().__init__(name=name)
    
        # Check network and image dimensions
        img_dims = config["img_dims"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]]) / 4))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        ndf = config["ndf"]
        num_layers = config["d_layers"]

        if config["d_time_layers"] is not None:
            self.time_layers = config["d_time_layers"]
        else:
           self.time_layers = []
       
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum numnber of discriminator layers: {max_num_layers}"
        self.conv_list = []

        # TODO: FIX
        # PixelGAN i.e. 1x1 receptive field
        if num_layers == 0:
            self.conv_list.append(
                GANDownBlock(
                    ndf, (1, 1, 1),
                    (1, 1, 1),
                    initialiser=initialiser,
                    batch_norm=False)) 
            
            self.conv_list.append(
                GANDownBlock(
                    ndf * 2,
                    (1, 1, 1),
                    (1, 1, 1),
                    initialiser=initialiser,
                    batch_norm=True))

            self.conv_list.append(tf.keras.layers.Conv3D(
                1, (1, 1, 1), (1, 1, 1),
                padding='same',
                kernel_initializer=initialiser))       

        # PatchGAN i.e. NxN receptive field
        else:
            batch_norm = False

            for i in range(0, num_layers):
                if i > 0: batch_norm = True
                channels = tf.minimum(ndf * 2 ** i, 512)

                if i > max_z_downsample:
                    strides = (2, 2, 1)
                    kernel = (4, 4, 2)
                else:
                    strides = (2, 2, 2)
                    kernel = (4, 4, 4)
                
                self.conv_list.append(
                    GANDownBlock(
                        channels,
                        kernel,
                        strides,
                        initialiser=initialiser,
                        model="discriminator",
                        batch_norm=batch_norm, name=f"down_{i}"))
            
            self.conv_list.append(tf.keras.layers.Conv3D(
                1, (4, 4, 1), (1, 1, 1),
                padding='valid',
                kernel_initializer=initialiser, name="output"))

        layer_names = [layer.name for layer in self.conv_list]

        for time_input in self.time_layers:
            assert time_input in layer_names, (time_input, layer_names)

    def build_model(self, x, t=None):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """

        return self(x, t).shape

    def call(self, x, t=None):

        for conv in self.conv_list:
            if conv.name in self.time_layers:
                x = conv(x, t, training=True)
            else:
                x = conv(x, training=True)

        return x
