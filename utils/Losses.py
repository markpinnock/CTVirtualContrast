import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def focused_mse(x, y, m):
    global_mse = tf.reduce_mean(tf.square(x - y))
    focal_mse = tf.reduce_mean(tf.square((x - y) * m))
    return global_mse, focal_mse

# TODO: allow e.g. MAE as well - consider keeping MSE/MAE within wrapper

class FocalLoss(keras.layers.Layer):
    def __init__(self, lambda_, name="focus_mse_loss"):
        super(FocalLoss, self).__init__(name=name)
        assert not lambda_ > 1.0, "Lambda must be in range [0, 1]"
        self.lambda_ = lambda_

    def call(self, y, x, mask):
        global_loss, focal_loss = focused_mse(x, y, mask)
        return (1 - self.lambda_) * global_loss + self.lambda_ * focal_loss


class FocalMetric(keras.metrics.Metric):
    def __init__(self, lambda_, name="focus_mse_metric"):
        super(FocalMetric, self).__init__(name=name)
        assert not lambda_ > 1.0, "Lambda must be in range [0, 1]"
        self.lambda_ = lambda_
        self.global_loss = self.add_weight(name="global", initializer="zeros")
        self.focal_loss = self.add_weight(name="focal", initializer="zeros")
    
    def update_state(self, y, x, mask):
        global_loss, focal_loss = focused_mse(x, y, mask)
        self.global_loss.assign_add(global_loss)
        self.focal_loss.assign_add(focal_loss)
    
    def result(self):
        return [tf.reduce_mean(self.global_loss), tf.reduce_mean(self.focal_loss)]

    def reset_states(self):
        self.global_loss.assign(0.0)
        self.focal_loss.assign(0.0)
