import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def focused_mse(a, b, m):
    global_mse = tf.reduce_mean(tf.square(a - b))
    focal_mse = tf.reduce_mean(tf.square((a - b) * m))
    return global_mse, focal_mse


class FocusedMSELoss(keras.layers.Layer):
    def __init__(self, lambda_, name="focus_mse_loss"):
        super(FocusedMSELoss, self).__init__(name=name)
        assert not lambda_ > 1.0, "Lambda must be in range [0, 1]"
        self.lambda_ = lambda_

    def call(self, pred, ace, mask):
        global_loss, focal_loss = focused_mse(pred, ace, mask)
        return (1 - self.lambda_) * global_loss + self.lambda_ * focal_loss
