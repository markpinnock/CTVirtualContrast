import tensorflow.keras as keras


class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = keras.layers.Conv3D(32, 3, activation='relu')
        self.conv2 = keras.layers.Conv3D(64, 3, activation='relu')
        self.tconv2 = keras.layers.Conv3DTranspose(32, 3, activation='relu')
        self.tconv1 = keras.layers.Conv3DTranspose(1, 3, activation='relu')
    
    def call(self, img):
        layer = self.conv1(img)
        layer = self.conv2(layer)
        layer = self.tconv2(layer)
        return self.tconv1(layer)