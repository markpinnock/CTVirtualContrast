import datetime
import matplotlib.pyplot as plt
import numpy as np
# import os
import sys
import tensorflow.keras as keras
import tensorflow as tf
import time

sys.path.append('..')
sys.path.append("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/scripts/")

from networks.GANWrapper import GAN
from networks.ResidualNet import ResNet
from networks.UNet import UNet
from utils.DataLoader import ImgLoader


def training_loop_UNet(epochs, phase, model, ds, show):
    ds_train, ds_val = ds
    start_time = time.time()

    for epoch in range(epochs):
        # model.metric[phase].reset_states()

        for data in ds_train:
            model.train_step(data, phase)

        print(f"Epoch {epoch + 1}, Loss {model.metric[phase].result()}")


    print(f"Time taken: {time.time() - start_time}")
    count = 0

    for data in ds_val:
        NCE, ACE, seg = data
        pred = model(NCE, phase, training=False).numpy()

        fig, axs = plt.subplots(2, 3)

        if phase == "seg":
            axs[0, 0].imshow(np.flipud(NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[0, 1].imshow(np.flipud(ACE[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[0, 2].imshow(np.flipud(pred[0, :, :, 0, 0]), cmap='hot', origin='lower')
            axs[1, 0].imshow(np.flipud(pred[0, :, :, 0, 0] * NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[1, 1].imshow(np.flipud(pred[0, :, :, 0, 0] * (NCE[0, :, :, 0, 0] - ACE[0, :, :, 0, 0])), cmap='gray', origin='lower')
            axs[1, 2].imshow(np.flipud(pred[0, :, :, 0, 0] - seg[0, :, :, 0, 0]), cmap='hot', origin='lower')
        
        elif phase == "vc":
            axs[0, 0].imshow(np.flipud(NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[0, 1].imshow(np.flipud(diff[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[0, 2].imshow(np.flipud(pred[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[1, 0].imshow(np.flipud(pred[0, :, :, 0, 0] + NCE[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[1, 1].imshow(np.flipud(pred[0, :, :, 0, 0] + NCE[0, :, :, 0, 0] - diff[0, :, :, 0, 0]), cmap='gray', origin='lower')
            axs[1, 2].imshow(np.flipud(pred[0, :, :, 0, 0] - diff[0, :, :, 0, 0]), cmap='gray', origin='lower')
        
        else:
            raise ValueError("seg or vc")

    if show:
        plt.show()
    else:
        plt.savefig(f"{SAVE_PATH}{phase}/{count:03d}.png", dpi=250)
        plt.close()
        count += 1


def training_loop_GAN(epochs, model, ds, show):
    ds_train, ds_val = ds
    start_time = time.time()

    for epoch in range(epochs):
        model.metric_dict["g_metric"].reset_states()
        model.metric_dict["d_metric_1"].reset_states()
        model.metric_dict["d_metric_2"].reset_states()
        model.L1metric.reset_states()

        for data in ds_train:
            NCE, ACE, mask = data
            model.train_step(NCE, ACE, mask)

        print(f"Epoch {epoch + 1}, G: {model.metric_dict['g_metric'].result():.4f} D1: {model.metric_dict['d_metric_1'].result():.4f}, D2: {model.metric_dict['d_metric_2'].result():.4f}, L1 [global focal]: {model.L1metric.result()}")

        for data in ds_val:
            NCE, ACE, seg = data
            pred = model.Generator(NCE, training=False).numpy()

            fig, axs = plt.subplots(2, 3)
            axs[0, 0].imshow(np.flipud(NCE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[0, 0].axis("off")
            axs[0, 1].imshow(np.flipud(ACE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[0, 1].axis("off")
            axs[0, 2].imshow(np.flipud(pred[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[0, 2].axis("off")
            axs[1, 0].imshow(np.flipud(seg[3, :, :, 11, 0], cmap='gray', origin='lower')
            axs[1, 0].axis("off")
            axs[1, 1].imshow(np.flipud(pred[3, :, :, 11, 0] + NCE[3, :, :, 11, 0] - ACE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[1, 1].axis("off")
            axs[1, 2].imshow(np.flipud(pred[3, :, :, 11, 0] - ACE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[1, 2].axis("off")

            if show:
                plt.show()
            else:
                plt.savefig(f"{SAVE_PATH}GAN/{epoch + 1}.png", dpi=250)
                plt.close()
            
            break

    print(f"Time taken: {time.time() - start_time}")


if __name__ == "__main__":

    """ Training script """

    SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/"
    FILE_PATH = "C:/ProjectImages/VirtualContrast/"

    # Hyperparameters
    MB_SIZE = 4
    NC = 8
    EPOCHS = 100
    ETA = 1e-4

    # Initialise datasets
    TrainGenerator = ImgLoader(
        file_path=FILE_PATH,
        dataset_type="training",
        num_folds=0,
        fold=0)

    ValGenerator = ImgLoader(
        file_path=FILE_PATH,
        dataset_type="validation",
        num_folds=0,
        fold=0)
    # TODO: convert to have one generator for train and val
    train_ds = tf.data.Dataset.from_generator(
        generator=TrainGenerator.data_generator,
        output_types=(tf.float32, tf.float32, tf.float32)
        ).batch(MB_SIZE)

    val_ds = tf.data.Dataset.from_generator(
        generator=ValGenerator.data_generator,
        output_types=(tf.float32, tf.float32, tf.float32)
        ).batch(MB_SIZE)

    # Compile model
    # Model = UNet(nc=NC, lambda_=0.0, optimiser=keras.optimizers.Adam(ETA))
    # Model = ResNet(nc=NC, optimiser=keras.optimizers.Adam(ETA))
    Model = GAN(
        g_optimiser=keras.optimizers.Adam(2e-4, 0.5, 0.999),
        d_optimiser=keras.optimizers.Adam(2e-4, 0.5, 0.999),
        lambda_=100)

    # curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/logs/" + curr_time
    # writer = tf.summary.create_file_writer(log_dir)

    # @tf.function
    # def trace(x):
    #     return Model(x)

    # tf.summary.trace_on(graph=True)
    # trace(tf.zeros((1, 128, 128, 12, 1)))
    # # print(Model.summary())


    # with writer.as_default():
    #     tf.summary.trace_export('graph', step=0)
    # exit()



    # Seg phase training loop
    # training_loop_UNet(EPOCHS, "seg", Model, (train_ds, val_ds))

    # Virtual contrast phase training loop
    # training_loop_UNet(EPOCHS, "vc", Model, (train_ds, val_ds))

    # GAN training loop
    training_loop_GAN(EPOCHS, Model, (train_ds, val_ds), False)
