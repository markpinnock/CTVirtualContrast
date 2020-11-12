import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def focused_mse(x, y, m):
    global_mse = tf.reduce_mean(tf.square(x - y) * (1 - m))
    focal_mse = tf.reduce_mean(tf.square(x - y) * m)
    return global_mse, focal_mse


@tf.function
def focused_mae(x, y, m):
    global_mae = tf.reduce_mean(tf.abs(x - y) * (1 - m))
    focal_mae = tf.reduce_mean(tf.abs(x - y) * m)
    return global_mae, focal_mae


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
    
    def result(self):
        return [tf.reduce_mean(self.global_loss), tf.reduce_mean(self.focal_loss)]

    def reset_states(self):
        self.global_loss.assign(0.0)
        self.focal_loss.assign(0.0)


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
