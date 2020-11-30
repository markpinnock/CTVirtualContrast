import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from networks.Layers import GANDownBlock, GANUpBlock


#-------------------------------------------------------------------------
""" PatchGAN discriminator for Pix2pix """

class Discriminator(keras.Model):

    """ Input:
        - initialiser e.g. keras.initializers.RandomNormal
        - nc: number of channels in first layer
        - num_layers: number of layers
        - img_dims: input image size
        Returns:
        - keras.Model """

    def __init__(
        self,
        initialiser,
        nc,
        num_layers,
        img_dim=(512, 512, 12), # TODO: make variable
        name=None):
        super(Discriminator, self).__init__(name=name)
    
        # Check network and image dimensions
        assert len(img_dim) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dim[0], img_dim[1]]) / 4))
        max_z_downsample = np.ceil(np.log2(img_dim[2]))
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum numnber of discriminator layers: {max_num_layers}"
        
        self.conv_list = []

        # TODO: FIX
        # PixelGAN i.e. 1x1 receptive field
        if num_layers == 0:
            self.conv_list.append(
                GANDownBlock(
                    nc, (1, 1, 1),
                    (1, 1, 1),
                    initialiser=initialiser,
                    batch_norm=False)) 
            
            self.conv_list.append(
                GANDownBlock(
                    nc * 2,
                    (1, 1, 1),
                    (1, 1, 1),
                    initialiser=initialiser,
                    batch_norm=True))

            self.conv_list.append(keras.layers.Conv3D(
                1, (1, 1, 1), (1, 1, 1),
                padding='same',
                kernel_initializer=initialiser))       

        # PatchGAN i.e. NxN receptive field
        else:
            batch_norm = False

            for i in range(0, num_layers):
                if i > 0: batch_norm = True
                channels = tf.minimum(nc * 2 ** i, 512)

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
            
            self.conv_list.append(keras.layers.Conv3D(
                1, (4, 4, 1), (1, 1, 1),
                padding='valid',
                kernel_initializer=initialiser, name="output"))

    def call(self, source, target, mask, test=False):
        # Test returns model parameters and feature map sizes
        if test: output_shapes = []
        x = keras.layers.concatenate([source, target, mask], axis=4, name="concat")

        for conv in self.conv_list:
            x = conv(x, training=True)
            if test: output_shapes.append(x.shape)
        
        if test:
            return output_shapes
        else:
            return x

#-------------------------------------------------------------------------
""" Generator for Pix2pix """

class Generator(keras.Model):

    """ Input:
            - initialiser e.g. keras.initializers.RandomNormal
            - nc: number of channels in first layer
            - num_layers: number of layers
            - img_dims: input image size
        Returns:
            - keras.Model """

    def __init__(
        self,
        initialiser,
        nc,
        num_layers,
        img_dim=(512, 512, 12),
        name=None):
        super(Generator, self).__init__(name=name)

        # Check network and image dimensions
        assert len(img_dim) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dim[0], img_dim[1]])))
        max_z_downsample = 2
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum numnber of discriminator layers: {max_num_layers}"
        
        self.encoder = []

        # Cache channels, strides and weights
        cache = {"channels": [], "strides": [], "kernels": []}

        for i in range(0, num_layers - 1):
            channels = np.min([nc * 2 ** i, 512])

            if i > max_z_downsample - 1:
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
            
            self.bottom_layer = keras.layers.Conv3D(
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

        self.final_layer = keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding='same', activation='tanh',
            kernel_initializer=initialiser, name="output")

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
