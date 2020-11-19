import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from networks.Layers import GANDownBlock, GANUpBlock


class Discriminator(keras.Model):
    def __init__(
        self,
        initialiser,
        nc,
        num_layers,
        img_dim=(512, 512, 12)):

        """ Implementation of PatchGAN discriminator
            Input:
                - initialiser e.g. keras.initializers.RandomNormal
                - nc: number of channels in first layer
                - num_layers: number of layers
                - img_dims: input image size """

        super(Discriminator, self).__init__()
    
        # Check network and image dimensions
        assert len(img_dim) == 3, "3D input only"
        max_num_layers = int(np.log2(np.min([img_dim[0], img_dim[1]]) / 4))
        max_z_downsample = np.ceil(np.log2(img_dim[2]))
        assert num_layers <= max_num_layers and num_layers >= 0, f"Maximum numnber of discriminator layers: {max_num_layers}"
        
        self.conv_list = []

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
                    initialiser=initialiser))
            
            self.conv_list.append(keras.layers.Conv3D(
                1, (1, 1, 1), (1, 1, 1),
                padding='same',
                kernel_initializer=initialiser))       

        # PatchGAN i.e. NxN receptive field
        else:
            self.conv_list.append(
                GANDownBlock(
                    nc, (4, 4, 2),
                    (2, 2, 2),
                    initialiser=initialiser,
                    batch_norm=False))

            for i in range(1, num_layers):
                channels = tf.minimum(nc * 2 ** i, 512)

                if i > max_z_downsample - 1:
                    z_stride = 1
                    z_weight = 1
                else:
                    z_stride = 2
                    z_weight = 2
                
                self.conv_list.append(
                    GANDownBlock(
                        channels,
                        (4, 4, z_weight),
                        (2, 2, z_stride),
                        initialiser=initialiser))
            
            self.conv_list.append(keras.layers.Conv3D(
                1, (4, 4, 1), (1, 1, 1),
                padding='valid',
                kernel_initializer=initialiser))

    def call(self, source, target, mask, test=False):
        # Test returns model parameters and feature map sizes
        if test: conv_shapes = []

        x = keras.layers.concatenate([source, target, mask], axis=4)

        for conv in self.conv_list:
            x = conv(x, training=True)

            if test: conv_shapes.append(x.shape)
        
        if test:
            return conv_shapes
        else:
            return x


class Generator(keras.Model):

    """ Implementation of generator
        Input:
            - initialiser e.g. keras.initializers.RandomNormal """

    def __init__(self, initialiser):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = GANDownBlock(64, (4, 4, 2), (2, 2, 2), initialiser=initialiser, batch_norm=False)
        self.conv2 = GANDownBlock(128, (4, 4, 2), (2, 2, 1), initialiser=initialiser)
        self.conv3 = GANDownBlock(256, (4, 4, 2), (2, 2, 2), initialiser=initialiser)
        self.conv4 = GANDownBlock(512, (4, 4, 2), (2, 2, 1), initialiser=initialiser)
        self.conv5 = GANDownBlock(512, (4, 4, 2), (2, 2, 1), initialiser=initialiser)
        self.conv6 = GANDownBlock(512, (4, 4, 2), (2, 2, 1), initialiser=initialiser)
        self.conv8 = keras.layers.Conv3D(
            512, (4, 4, 2), (2, 2, 1),
            padding="same", activation="relu",
            kernel_initializer=initialiser)

        # Decoder
        self.tconv1 = GANUpBlock(512, (2, 2, 1), initialiser=initialiser, dropout=True)
        self.tconv2 = GANUpBlock(512, (2, 2, 1), initialiser=initialiser, dropout=True)
        self.tconv3 = GANUpBlock(512, (2, 2, 1), initialiser=initialiser, dropout=True)
        self.tconv4 = GANUpBlock(256, (2, 2, 1), initialiser=initialiser)
        self.tconv6 = GANUpBlock(128, (2, 2, 2), initialiser=initialiser)
        self.tconv7 = GANUpBlock(64, (2, 2, 1), initialiser=initialiser)
        self.tconv8 = keras.layers.Conv3DTranspose(
            1, (4, 4, 4), (2, 2, 2),
            padding='same', activation='tanh',
            kernel_initializer=initialiser)

    def call(self, x, test=False):
        # Encode 128 128 12
        dn1 = self.conv1(x, training=True) # 64 64 6
        dn2 = self.conv2(dn1, training=True) # 32 32 6
        dn3 = self.conv3(dn2, training=True) # 16 16 3
        dn4 = self.conv4(dn3, training=True) # 8 8 3
        dn5 = self.conv5(dn4, training=True) # 4 4 3
        dn6 = self.conv6(dn5, training=True) # 2 2 3
        dn8 = self.conv8(dn6) # 1 1 3

        # Decode
        up1 = self.tconv1(dn8, dn6, training=True) # 2 2 3
        up2 = self.tconv2(up1, dn5, training=True) # 4 4 3
        up3 = self.tconv3(up2, dn4, training=True) # 8 8 3
        up4 = self.tconv4(up3, dn3, training=True) # 16 16 3
        up5 = self.tconv6(up4, dn2, training=True) # 32 32 6
        up7 = self.tconv7(up5, dn1, training=True) # 64 64 6
        up8 = self.tconv8(up7) # 128 128 12
        
        if test:
            output_shapes = [dn1.shape, dn2.shape, dn3.shape, dn4.shape, dn5.shape, dn6.shape, dn8.shape, up1.shape, up2.shape, up3.shape, up4.shape, up5.shape, up7.shape, up8.shape]
            return output_shapes

        return up8