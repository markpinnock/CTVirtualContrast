import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")

from networks.Layers import DownBlock


class ResNet(keras.Model):
    def __init__(self, nc, optimiser):
        super(ResNet, self).__init__()
        self.optimiser = optimiser
        self.loss = keras.losses.MeanSquaredError()
        self.metric = keras.metrics.MeanSquaredError()

        self.down1 = DownBlock(nc, (1, 1, 1))
        self.down2 = DownBlock(nc * 2, (1, 1, 1))
        self.down3 = DownBlock(nc * 4, (1, 1, 1))
        # self.down4 = DownBlock(nc * 8, (1, 1, 1))
        self.out = keras.layers.Conv3D(1, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='linear')
    
    def call(self, x):
        h, _ = self.down1(x)
        h, _ = self.down2(h)
        h, _ = self.down3(h)
        # x, _ = self.down4(x)
        return self.out(h)

    def train_step(self, data):
        imgs, labels = data

        with tf.GradientTape() as tape:
            predictions = self(imgs, training=True)
            loss = self.loss(labels, predictions)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))
        self.metric.update_state(labels, predictions)
    
    def test_step(self, data):
        imgs, labels = data
        predictions = self(imgs, training=False)
        self.metric.update_state(labels, predictions)