import tensorflow as tf


class HyperNetwork_v02(tf.keras.layers.Layer):

    def __init__(self, kernel_dims, name: str = None):
        super().__init__(name)
        self.kernel_dims = kernel_dims
        self.shared_layers = []
        self.concat = tf.keras.layers.Concatenate()

        for i in range(2):
            self.shared_layers.append(tf.keras.layers.Dense(64, kernel_initializer="he_normal", activation="relu", name=f"shared_{i}"))
    
    def build(self, input_shape):
        super().build(input_shape)
        self.layer_W = {
            layer: self.add_weight(
                name=layer, shape=[64, tf.reduce_prod(shape)], initializer="glorot_normal", trainable=True
                ) for layer, shape in self.kernel_dims.items()
        }

        self.layer_b = {
            layer: self.add_weight(
                name=layer, shape=[tf.reduce_prod(shape)], initializer="zeros", trainable=True
                ) for layer, shape in self.kernel_dims.items()
        }

    def call(self, source_time=None, target_time=None):
        if source_time is None:
            x = tf.ones([1, 2])
        else:
            x = self.concat([source_time[:, tf.newaxis], target_time[:, tf.newaxis]])
        
        for dense in self.shared_layers:
            x = dense(x)
        
        layer_weights = {}

        for layer, shape in self.kernel_dims.items():
            kernel = self.layer_W[layer]
            bias = self.layer_b[layer]
            K = tf.add(tf.matmul(x, kernel), bias)
            layer_weights[layer] = tf.reshape(K, shape)

        return layer_weights
