import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def trainStep(CE, NCE, model, optimiser, train_metric):
    loss_func = keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        prediction = model(NCE, training=True)
        curr_loss = loss_func(CE, prediction)

    gradients = tape.gradient(curr_loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric(CE, prediction)


@tf.function
def valStep(CE, NCE, model, val_metric):
    prediction = model(NCE, training=False)
    val_metric(CE, prediction)