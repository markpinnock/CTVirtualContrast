import tensorflow as tf
import tensorflow.keras as keras


""" Loss functions used in Version 02 - standard GAN losses """

#-------------------------------------------------------------------------
""" Non-saturating minimax loss
    Goodfellow et al. Generative adversarial networks. NeurIPS, 2014
    https://arxiv.org/abs/1406.2661 """

@tf.function
def minimax_D(real_output, fake_output):
    real_loss = keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
    fake_loss = keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    return real_loss + fake_loss

@tf.function
def minimax_G(fake_output):
    fake_loss = keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output, from_logits=True)
    return fake_loss


#-------------------------------------------------------------------------
""" Pix2pix L1 loss
    Isola et al. Image-to-image translation with conditional adversarial networks.
    Proceedings of the IEEE conference on computer vision and pattern recognition, 2017.
    https://arxiv.org/abs/1406.2661 """

@tf.function
def L1(real_img, fake_img):
    return tf.reduce_mean(tf.abs(real_img - fake_img))


#-------------------------------------------------------------------------
""" Wasserstein loss
    Arjovsky et al. Wasserstein generative adversarial networks.
    International conference on machine learning. PMLR, 2017
    https://arxiv.org/abs/1701.07875 """

@tf.function
def wasserstein_D(real_output, fake_output):
    return tf.reduce_mean(fake_output - real_output)

@tf.function
def wasserstein_G(fake_output):
    return tf.reduce_mean(-fake_output)


#-------------------------------------------------------------------------
""" Wasserstein loss gradient penalty
    Gulrajani et al. Improved training of Wasserstein GANs. NeurIPS, 2017
    https://arxiv.org/abs/1704.00028 """

@tf.function
def gradient_penalty(real_img, fake_img, D, scale):
    epsilon = tf.random.uniform([fake_img.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = (epsilon * real_img) + ((1 - epsilon) * fake_img)

    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        D_hat = D(x_hat, scale, training=True)
    
    gradients = tape.gradient(D_hat, x_hat)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=(1, 2)) + 1e-8)
    grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1))

    return grad_penalty
