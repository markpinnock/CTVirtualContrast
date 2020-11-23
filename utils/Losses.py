import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def focused_mse(x, y, m):
    global_squared_err = tf.reshape(tf.square(x - y) * (1 - m), (x.shape[0], -1))
    focal_squared_err = tf.reshape(tf.square(x - y) * m, (x.shape[0], -1))
    flat_m = tf.reshape(m, (x.shape[0], -1))
    global_mse = tf.reduce_sum(global_squared_err, -1) / (tf.reduce_sum(1 - flat_m, -1) + 1e-12)
    focal_mse = tf.reduce_sum(focal_squared_err, -1) / (tf.reduce_sum(flat_m, -1) + 1e-12)
    mb_global_mse = tf.reduce_sum(global_mse)
    mb_focal_mse = tf.reduce_sum(focal_mse)
    
    return mb_global_mse, mb_focal_mse


@tf.function
def focused_mae(x, y, m):
    global_absolute_err = tf.reshape(tf.abs(x - y) * (1 - m), (x.shape[0], -1))
    focal_absolute_err = tf.reshape(tf.abs(x - y) * m, (x.shape[0], -1))
    flat_m = tf.reshape(m, (x.shape[0], -1))
    global_mae = tf.reduce_sum(global_absolute_err, -1) / (tf.reduce_sum(1 - flat_m, -1) + 1e-12)
    focal_mae = tf.reduce_sum(focal_absolute_err, -1) / (tf.reduce_sum(flat_m, -1) + 1e-12)
    mb_global_mae = tf.reduce_sum(global_mae)
    mb_focal_mae = tf.reduce_sum(focal_mae)

    return mb_global_mae, mb_focal_mae


class FocalLoss(keras.layers.Layer):
    def __init__(self, mu, loss_fn, name="focus_loss"):
        super(FocalLoss, self).__init__(name=name)
        assert not mu > 1.0, "Lambda must be in range [0, 1]"
        self.mu = mu

        if loss_fn == "mse":
            self.focal = focused_mse
        elif loss_fn == "mae":
            self.focal = focused_mae
        else:
            raise ValueError("'mse' or 'mae'")

    def call(self, y, x, mask):
        global_loss, focal_loss = self.focal(x, y, mask)
        return (1 - self.mu) * global_loss + self.mu * focal_loss


class FocalMetric(keras.metrics.Metric):
    def __init__(self, loss_fn, name="focus_metric"):
        super(FocalMetric, self).__init__(name=name)
        self.global_loss = self.add_weight(name="global", initializer="zeros")
        self.focal_loss = self.add_weight(name="focal", initializer="zeros")
        self.N = self.add_weight(name="N", initializer="zeros")
        
        if loss_fn == "mse":
            self.focal = focused_mse
        elif loss_fn == "mae":
            self.focal = focused_mae
        else:
            raise ValueError("'mse' or 'mae'") 

    def update_state(self, y, x, mask):
        global_loss, focal_loss = self.focal(x, y, mask)
        self.global_loss.assign_add(global_loss)
        self.focal_loss.assign_add(focal_loss)
        self.N.assign_add(x.shape[0])
    
    def result(self):
        return [self.global_loss / self.N, self.focal_loss / self.N]

    def reset_states(self):
        self.global_loss.assign(0.0)
        self.focal_loss.assign(0.0)
        self.N.assign(1e-12) # So no nans if result called after reset_states


class DiceLoss(keras.losses.Loss):
    def __init__(self, name="dice_loss"):
        super(DiceLoss, self).__init__(name=name)

    def call(self, x, y):
        numer = tf.reduce_sum(x * y, axis=[1, 2, 3, 4]) * 2
        denom = tf.reduce_sum(x, axis=[1, 2, 3, 4]) + tf.reduce_sum(y, axis=[1, 2, 3, 4]) + 1e-6
        dice = tf.reduce_mean(numer / denom)
        return 1 - dice


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
