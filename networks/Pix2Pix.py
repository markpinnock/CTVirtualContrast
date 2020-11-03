import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from networks.Layers import GANDownBlock, GANUpBlock


class Discriminator(keras.Model):
    def __init__(self, initialiser):

        """ Implementation of PatchGAN discriminator
            Input:
                - initialiser e.g. keras.initializers.RandomNormal """

        super(Discriminator, self).__init__()
        # TODO: make disc arbitrary patch size
        self.conv1 = GANDownBlock(64, (2, 2, 2), initialiser=initialiser, batch_norm=False)
        self.conv2 = GANDownBlock(128, (2, 2, 1), initialiser=initialiser)
        self.conv3 = GANDownBlock(256, (2, 2, 2), initialiser=initialiser)
        self.conv4 = GANDownBlock(512, (2, 2, 1), initialiser=initialiser)
        self.conv5 = GANDownBlock(512, (2, 2, 2), initialiser=initialiser)
        self.conv6 = keras.layers.Conv3D(
            1, (4, 4, 1), (1, 1, 1),
            padding='same',
            kernel_initializer=initialiser)

    def call(self, source, target, training):
        # Input 128 128 12
        x = keras.layers.concatenate([source, target], axis=4)
        dn1 = self.conv1(x) # 64 64 6
        dn2 = self.conv2(dn1, training) # 32 32 6
        dn3 = self.conv3(dn2, training) # 16 16 3
        dn4 = self.conv4(dn3, training) # 8 8 3
        dn5 = self.conv5(dn4, training) # 4 4 1
        dn6 = self.conv6(dn5)
        
        return dn6


class Generator(keras.Model):

    """ Implementation of generator
        Input:
            - initialiser e.g. keras.initializers.RandomNormal """

    def __init__(self, initialiser):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = GANDownBlock(64, (2, 2, 2), initialiser=initialiser, batch_norm=False)
        self.conv2 = GANDownBlock(128, (2, 2, 1), initialiser=initialiser)
        self.conv3 = GANDownBlock(256, (2, 2, 2), initialiser=initialiser)
        self.conv4 = GANDownBlock(512, (2, 2, 1), initialiser=initialiser)
        self.conv5 = GANDownBlock(512, (2, 2, 1), initialiser=initialiser)
        self.conv6 = GANDownBlock(512, (2, 2, 1), initialiser=initialiser)
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

    def call(self, x, training):
        # Encode 128 128 12
        dn1 = self.conv1(x) # 64 64 6
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

        return up8