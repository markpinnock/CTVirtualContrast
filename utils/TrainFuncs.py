import tensorflow as tf


@tf.function
def trainStep(CE, NCE, model, optimiser, loss, train_metric):
    with tf.GradientTape() as tape:
        prediction = model(NCE)
        curr_loss = loss(CE, prediction)
    
    gradients = tape.gradient(curr_loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric(CE, prediction)


@tf.function
def valStep(CE, NCE, model, val_metric):
    prediction = model(NCE)
    val_metric(CE, prediction)