import numpy as np
import sys
import tensorflow as tf

from .layers import GANDownBlock, GANUpBlock


#-------------------------------------------------------------------------
""" PatchGAN discriminator for Pix2pix """

class Discriminator(tf.keras.Model):

    """ Input:
        - initialiser: e.g. keras.initializers.RandomNormal
        - config: configuration json
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, d_focal=False, name=None):
        super(Discriminator, self).__init__(name=name)
    
        # Check network and image dimensions
        img_dims = config["DATA"]["IMG_DIMS"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]]) / 4))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))

        # if d_focal:
        #     ndf = config["HYPERPARAMS"]["NDF_F"]
        #     num_layers = config["HYPERPARAMS"]["D_LAYERS_F"]
        # else:
        ndf = config["HYPERPARAMS"]["NDF"]
        num_layers = config["HYPERPARAMS"]["D_LAYERS"]
       
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
                    kernel = (4, 4, 1)
                else:
                    strides = (2, 2, 2)
                    kernel = (4, 4, 2)
                
                self.conv_list.append(
                    GANDownBlock(
                        channels,
                        kernel,
                        strides,
                        initialiser=initialiser,
                        batch_norm=batch_norm, name=f"downblock_{i}"))
            
            self.conv_list.append(tf.keras.layers.Conv3D(
                1, (4, 4, 1), (1, 1, 1),
                padding='valid',
                kernel_initializer=initialiser, name="output"))

    def build_model(self, x):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """

        return self(x).shape

    def call(self, x, test=False):
        # Test returns model parameters and feature map sizes
        if test: output_shapes = []

        for conv in self.conv_list:
            x = conv(x, training=True)
            if test: output_shapes.append(x.shape)
        
        if test:
            return output_shapes
        else:
            return x

#-------------------------------------------------------------------------
""" Generator for Pix2pix """

class Generator(tf.keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(self, initialiser, config, name=None):
        super(Generator, self).__init__(name=name)

        # Check network and image dimensions
        img_dims = config["DATA"]["IMG_DIMS"]
        assert len(img_dims) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dims[0], img_dims[1]])))
        max_z_downsample = int(np.floor(np.log2(img_dims[2])))
        ngf = config["HYPERPARAMS"]["NGF"]
        num_layers = config["HYPERPARAMS"]["G_LAYERS"]
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum number of discriminator layers: {max_num_layers}"
        self.encoder = []

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, num_layers - 1):
            channels = np.min([ngf * 2 ** i, 512])

            if i >= max_z_downsample - 1:
                strides = (2, 2, 1)
                kernel = (4, 4, 1)
            else:
                strides = (2, 2, 2)
                kernel = (4, 4, 2)

            cache["channels"].append(channels)
            cache["strides"].append(strides)
            cache["kernels"].append(kernel)

            self.encoder.append(
                GANDownBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    batch_norm=True, name=f"down_{i}"))
        
        # TODO: Needs better implementation to avoid concat bug from above loop
        self.bottom_layer = tf.keras.layers.Conv3D(
                channels, kernel, strides,
                padding="same", activation="relu",
                kernel_initializer=initialiser, name="bottom")

        cache["strides"].append(strides)
        cache["kernels"].append(kernel)

        cache["channels"].reverse()
        cache["kernels"].reverse()
        cache["strides"].reverse()

        self.decoder = []
        dropout = True

        for i in range(0, num_layers - 1):
            if i > 2: dropout = False
            channels = cache["channels"][i]
            strides = cache["strides"][i]
            kernel = cache["kernels"][i]
            
            self.decoder.append(
                GANUpBlock(
                    channels,
                    kernel,
                    strides,
                    initialiser=initialiser,
                    batch_norm=True,
                    dropout=dropout, name=f"up_block{i}"))

        self.final_layer = tf.keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding='same', activation='tanh',
            kernel_initializer=initialiser, name="output")

    def build_model(self, x):

        """ Build method takes tf.zeros((input_dims)) and returns
            shape of output - all layers implicitly built and weights set to trainable """
        
        return self(x).shape

    def call(self, x, test=False):
        skip_layers = []
        if test: output_shapes = []

        for conv in self.encoder:
            x = conv(x, training=True)
            skip_layers.append(x)
            if test: output_shapes.append(x.shape)
        
        x = self.bottom_layer(x, training=True)
        if test: output_shapes.append(x.shape)
        skip_layers.reverse()

        for skip, tconv in zip(skip_layers, self.decoder):
            x = tconv(x, skip, training=True)
            if test: output_shapes.append(x.shape)
        
        x = self.final_layer(x, training=True)
        if test: output_shapes.append(x.shape)
        
        if test:
            return output_shapes
        else:
            return x
