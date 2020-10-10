import tensorflow as tf
import tensorflow.keras as keras


class DownBlock(keras.layers.Layer):
    def __init__(self, nc, pool_strides):
        super(DownBlock, self).__init__(self)

        self.conv1 = keras.layers.Conv3D(nc, (4, 4, 2), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv3D(nc, (4, 4, 2), strides=(1, 1, 1), padding='same', activation='relu')
        self.pool = keras.layers.MaxPool3D((2, 2, 2), strides=pool_strides, padding='same')
        # Consider pool -> conv3
        # Consider dropout, normalisation
        # Leaky ReLU
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class UpBlock(keras.layers.Layer):
    def __init__(self, nc, tconv_strides):
        super(UpBlock, self).__init__(self)

        self.tconv = keras.layers.Conv3DTranspose(nc, (4, 4, 2), strides=tconv_strides, padding='same', activation='relu')
        self.conv1 = keras.layers.Conv3D(nc, (4, 4, 2), strides=(1, 1, 1), padding='same', activation='relu')
        self.conv2 = keras.layers.Conv3D(nc, (4, 4, 2), strides=(1, 1, 1), padding='same', activation='relu')
    
    def call(self, x, skip):
        x = self.tconv(x)
        x = keras.layers.concatenate([x, skip], axis=4)
        x = self.conv1(x)
        return self.conv2(x)


class Discriminator(keras.Model):
    def __init__(self, image_dims):
        super(Discriminator, self).__init__()
        
        self.conv1 = DownBlock(64, (2, 2, 2)) # 128 128 6
        self.conv2 = DownBlock(128, (2, 2, 1)) # 64 64 6
        self.conv3 = DownBlock(256, (2, 2, 2)) # 32 32 3
        self.conv4 = DownBlock(512, (2, 2, 1)) # 16 16 3
        self.conv5 = DownBlock(512, (2, 2, 2)) # 8 8 1
        self.conv6 = DownBlock(512, (2, 2, 1)) # 4 4 1
        self.conv7 = keras.layers.Conv3D(1, (4, 4, 1), strides=(1, 1, 1), padding='VALID', activation='linear')(dconv6)
        # Convert to Leaky ReLU
    def call(self, fake, real):
        concat = keras.layers.concatenate([fake, real], axis=4)
        conv1 = self.conv1(concat)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
    
        return self.conv7(conv6)


def generatorModel(input_dims):
    # Encoder
    self.dn1 = DownBlock(64, (2, 2, 2)) # 256 256 6
    self.dn2 = DownBlock(128, (2, 2, 1)) 
    self.dn3 = DownBlock(256, (2, 2, 2)) # 64 64 3
    self.dn4 = DownBlock(512, (2, 2, 1))
    self.dn5 = DownBlock(512, (2, 2, 1)) # 16 16 3
    self.dn6 = DownBlock(512, dn5, (2, 2, 1)) # 8 8 3

    # Decoder
    self.up1 = UpBlock(512, (2, 2, 1))
    self.up2 = upBlock(512, (2, 2, 1))
    self.up3 = upBlock(256, (2, 2, 1)) 
    self.up4 = upBlock(128, (2, 2, 2))
    self.up5 = upBlock(64, (2, 2, 1))
    self.out = keras.layers.Conv3DTranspose(1, (4, 4, 4), strides=(2, 2, 2), padding='same', activation='tanh')(up1)

    def call(img):
        # Encode
        dn1 = self.dn1(img)
        dn2 = self.dn1(dn1)
        dn3 = self.dn1(dn2)
        dn4 = self.dn1(dn3)
        dn5 = self.dn1(dn4)
        dn6 = self.dn1(dn5)

        # Decode
        up1 = self.up1(dn6, dn5)
        up2 = self.up1(up1, dn4)
        up3 = self.up1(up2, dn3)
        up4 = self.up1(up3, dn2)
        up5 = self.up1(up4, dn1)

        return self.up6(up5)