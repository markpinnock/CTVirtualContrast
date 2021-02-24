import tensorflow as tf
import tensorflow.keras as keras


#-------------------------------------------------------------------------
""" Standard L2 loss, allows label weighting based on RBF etc. """

@tf.function
def mse(x, y, w):
    squared_err = tf.square(x - y)
    mse = tf.reduce_mean(squared_err, axis=[1, 2, 3, 4])
    mb_mse = tf.reduce_sum(mse * w)
    
    return mb_mse

#-------------------------------------------------------------------------
""" Standard L1 loss, allows label weighting based on RBF etc. """

@tf.function
def mae(x, y, w):
    absolute_err = tf.abs(x - y)
    mae = tf.reduce_mean(absolute_err, axis=[1, 2, 3, 4])
    mb_mae = tf.reduce_sum(mae * w)

    return mb_mae

#-------------------------------------------------------------------------
""" Focused L2 loss, calculates L2 inside and outside masked area """

@tf.function
def focused_mse(x, y, m, w):
    global_squared_err = tf.reshape(tf.square(x - y) * (1 - m), (x.shape[0], -1))
    focal_squared_err = tf.reshape(tf.square(x - y) * m, (x.shape[0], -1))
    flat_m = tf.reshape(m, (x.shape[0], -1))
    global_mse = tf.reduce_sum(global_squared_err, -1) / (tf.reduce_sum(1 - flat_m, -1) + 1e-12)
    focal_mse = tf.reduce_sum(focal_squared_err, -1) / (tf.reduce_sum(flat_m, -1) + 1e-12)
    
    mb_global_mse = tf.reduce_sum(global_mse * w)
    mb_focal_mse = tf.reduce_sum(focal_mse * w)
    
    return mb_global_mse, mb_focal_mse

#-------------------------------------------------------------------------
""" Focused L1 loss, calculates L1 inside and outside masked area """

@tf.function
def focused_mae(x, y, m, w):
    global_absolute_err = tf.reshape(tf.abs(x - y) * (1 - m), (x.shape[0], -1))
    focal_absolute_err = tf.reshape(tf.abs(x - y) * m, (x.shape[0], -1))
    flat_m = tf.reshape(m, (x.shape[0], -1))
    global_mae = tf.reduce_sum(global_absolute_err, -1) / (tf.reduce_sum(1 - flat_m, -1) + 1e-12)
    focal_mae = tf.reduce_sum(focal_absolute_err, -1) / (tf.reduce_sum(flat_m, -1) + 1e-12)

    mb_global_mae = tf.reduce_sum(global_mae * w)
    mb_focal_mae = tf.reduce_sum(focal_mae * w)

    return mb_global_mae, mb_focal_mae

#-------------------------------------------------------------------------
""" Focal loss, weights loss according to masked area """

class Loss(keras.layers.Layer):
    def __init__(self, config, loss_fn, name=None):
        super().__init__(name=name)
        assert not config["MU"] > 1.0, "Mu must be in range [0, 1]"
        self.mu = config["MU"]
        self.loss_fn = loss_fn
        self.gamma = config["GAMMA"]
        print("==================================================")
        print(f"Loss using mu {self.mu}, loss {loss_fn}, gamma {self.gamma}")

        if loss_fn == "mse":
            if self.mu:
                self.loss = focused_mse
            else:
                self.loss = mse

        elif loss_fn == "mae":
            if self.mu:
                self.loss = focused_mae

            else:
                self.loss = mae

        else:
            raise ValueError("'mse' or 'mae'")

        if self.gamma:
            self.label_weighting = calc_RBF
        else:
            self.label_weighting = None

    def call(self, y, x, mask):
        label_weights = calc_RBF(x, y, self.gamma)

        if self.mu:
            global_loss, focal_loss = self.loss(x, y, mask, label_weights)
            return (1 - self.mu) * global_loss + self.mu * focal_loss

        else:
            return self.loss(x, y, label_weights)

#-------------------------------------------------------------------------
""" Focal metric, weights loss according to masked area """

class FocalMetric(keras.metrics.Metric):
    def __init__(self, config, loss_fn, name=None):
        super().__init__(name=name)
        self.global_loss = self.add_weight(name="global", initializer="zeros")
        self.focal_loss = self.add_weight(name="focal", initializer="zeros")
        self.label_weights = self.add_weight(name="weights", initializer="zeros")
        self.N = self.add_weight(name="N", initializer="zeros")

        if loss_fn == "mse":
            self.focal = focused_mse
        elif loss_fn == "mae":
            self.focal = focused_mae
        else:
            raise ValueError("'mse' or 'mae'")
        
        self.label_weighting = calc_RBF

        if config["GAMMA"]:
            self.gamma = config["GAMMA"]
        else:
            self.gamma = 1e-3

    def update_state(self, y, x, mask):
        weights = calc_RBF(x, y, self.gamma)
        global_loss, focal_loss = self.focal(x, y, mask, weights)
        self.global_loss.assign_add(global_loss)
        self.focal_loss.assign_add(focal_loss)
        self.label_weights.assign_add(tf.reduce_sum(weights))
        self.N.assign_add(x.shape[0])
    
    def result(self):
        return [self.global_loss / self.N, self.focal_loss / self.N, self.label_weights / self.N]

    def reset_states(self):
        self.global_loss.assign(0.0)
        self.focal_loss.assign(0.0)
        self.label_weights.assign(0.0)
        self.N.assign(1e-12) # So no nans if result called after reset_states

#-------------------------------------------------------------------------
""" Dice coefficient-based loss """
# TODO: FIX

class DiceLoss(keras.losses.Loss):
    def __init__(self, name=None):
        super(DiceLoss, self).__init__(name=name)

    def call(self, x, y):
        numer = tf.reduce_sum(x * y, axis=[1, 2, 3, 4]) * 2
        denom = tf.reduce_sum(x, axis=[1, 2, 3, 4]) + tf.reduce_sum(y, axis=[1, 2, 3, 4]) + 1e-6
        dice = tf.reduce_mean(numer / denom)
        return 1 - dice

#-------------------------------------------------------------------------
""" Dice coefficient-based metric """
# TODO: FIX

class DiceMetric(keras.metrics.Metric):
    def __init__(self, name="dice_metric"):
        super(DiceMetric, self).__init__(name=name)
        self.loss = self.add_weight(name="dice", initializer="zeros")
    
    def update_state(self, x, y):
        numer = tf.reduce_sum(x * y, axis=[1, 2, 3, 4]) * 2
        denom = tf.reduce_sum(x, axis=[1, 2, 3, 4]) + tf.reduce_sum(y, axis=[1, 2, 3, 4]) + 1e-6
        dice = tf.reduce_mean(numer / denom)
        self.loss.assign_add(1 - dice)

    def result(self):
        return tf.reduce_mean(self.loss)

    def reset_states(self):
        self.loss.assign(0.0)

#-------------------------------------------------------------------------
""" Normalised cross-correlation """

def calc_NCC(a, b):
    N = tf.cast(tf.reduce_prod(a.shape[1:]), tf.float32)
    mu_a = tf.reduce_mean(a, axis=[1, 2, 3, 4])
    mu_b = tf.reduce_mean(b, axis=[1, 2, 3, 4])
    sig_a = tf.math.reduce_std(a, axis=[1, 2, 3, 4])
    sig_b = tf.math.reduce_std(b, axis=[1, 2, 3, 4])

    return tf.reduce_mean((a - mu_a) * (b - mu_b), axis=[1, 2, 3, 4]) / (sig_a * sig_b + 1e-12)

#-------------------------------------------------------------------------
""" Radial basis function """

def calc_RBF(a, b, gamma=1e-3):
    K =  tf.exp(-gamma * tf.reduce_sum(tf.pow(a - b, 2), axis=[1, 2, 3, 4]))
    return K / (tf.reduce_sum(K) + 1e-8)

#-------------------------------------------------------------------------

if __name__ == "__main__":
    a = tf.ones((2, 4, 4, 1, 1))
    b = tf.zeros((2, 4, 4, 1, 1))
    x = tf.concat([a, b], axis=3)
    y = tf.concat([b, a], axis=3)
    print(calc_NCC(x, y).numpy(), calc_RBF(x, y, 1).numpy())
    print(calc_NCC(x, x).numpy(), calc_RBF(x, x, 1).numpy())
    print(calc_NCC(y, x).numpy(), calc_RBF(x, y, 1).numpy())
    print(calc_NCC(y, y).numpy(), calc_RBF(y, y, 1).numpy())
    print("=======================")
    import numpy as np
    img = np.load("C:/ProjectImages/VirtualContrast/AC/T002R1AC007_000.npy")
    import matplotlib.pyplot as plt
    i1 = tf.constant(np.fliplr(img), tf.float32)[tf.newaxis, :, :, :, tf.newaxis]
    i2 = tf.constant(img, tf.float32)[tf.newaxis, :, :, :, tf.newaxis]
    print(calc_NCC(i1, tf.zeros(i1.shape, tf.float32)).numpy(), calc_RBF(i1, tf.zeros(i1.shape, tf.float32), 1).numpy())
    print(calc_NCC(i1, i1).numpy(), calc_RBF(i1, i1, 1).numpy())
    print(calc_NCC(i1, i2).numpy(), calc_RBF(i1, i2, 1).numpy())
