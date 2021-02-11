import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time


#-------------------------------------------------------------------------
""" Function to print summary of model e.g. layers, feature map dims """

def print_model_summary(model, config, model_type):
    print("===================================")
    print(model.name)
    print("===================================")

    weight_dict = {}
    total_weights = 0
    if model_type == "G":
        in_ch = 1
    else:
        in_ch = config["HYPERPARAMS"]["D_IN_CH"]

    xy_dims = np.array(config["EXPT"]["IMG_DIMS"])[0:2] // config["EXPT"]["DOWN_SAMP"]
    z_dims = np.array(config["EXPT"]["IMG_DIMS"])[2]
    x = np.zeros([1, xy_dims[0], xy_dims[1], z_dims, in_ch], dtype=np.float32)
    output_shapes = model(x, test=True)

    for layer in model.layers:
        weight_dict[layer.name] = []

        for weight in layer.trainable_weights:
            total_weights += np.prod(weight.shape.as_list())
            weight_dict[layer.name].append(weight.shape.as_list())
    
    assert len(output_shapes) == len(weight_dict), "Weight list and output shape list do not match"

    for idx, layer in enumerate(model.layers):
        print(f"{layer.name}: {weight_dict[layer.name]} {output_shapes[idx]}")

    print("===================================")
    print(f"Total weights: {total_weights}")
    print("===================================")

#-------------------------------------------------------------------------
""" UNet training loop """

def training_loop_UNet(config, phase, model, ds):

    """ config: configuration json 
        phase: for seg -> VC training
        model: expects BaseGAN or subclasses (keras.Model)
        ds: Tensorflow dataset """

    EPOCHS = config["EPOCHS"]
    SAVE_PATH = config["SAVE_PATH"]

    if not os.path.exists(f"{SAVE_PATH}seg/"):
        os.mkdir(f"{SAVE_PATH}seg/")
    if not os.path.exists(f"{SAVE_PATH}vc/"):
        os.mkdir(f"{SAVE_PATH}vc/")

    ds_train, ds_val = ds
    start_time = time.time()

    for epoch in range(EPOCHS):
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
            axs[0, 0].imshow(NCE[0, :, :, 11, 0].T, cmap="gray")
            axs[0, 1].imshow(ACE[0, :, :, 11, 0].T, cmap="gray")
            axs[0, 2].imshow(pred[0, :, :, 11, 0].T, cmap="hot")
            axs[1, 0].imshow(pred[0, :, :, 11, 0].T * NCE[0, :, :, 11, 0].T, cmap="gray")
            axs[1, 1].imshow(pred[0, :, :, 11, 0].T * NCE[0, :, :, 11, 0].T - ACE[0, :, :, 11, 0].T, cmap="gray")
            axs[1, 2].imshow(pred[0, :, :, 11, 0].T - seg[0, :, :, 11, 0].T, cmap="hot")
        
        elif phase == "vc":
            axs[0, 0].imshow(NCE[0, :, :, 11, 0].T, cmap="gray")
            axs[0, 1].imshow(diff[0, :, :, 11, 0].T, cmap="gray")
            axs[0, 2].imshow(pred[0, :, :, 11, 0].T, cmap="gray")
            axs[1, 0].imshow(pred[0, :, :, 11, 0].T + NCE[0, :, :, 11, 0].T, cmap="gray")
            axs[1, 1].imshow(pred[0, :, :, 11, 0].T + NCE[0, :, :, 11, 0].T - diff[0, :, :, 0, 0].T, cmap="gray")
            axs[1, 2].imshow(pred[0, :, :, 11, 0].T - diff[0, :, :, 11, 0].T, cmap="gray")
        
        else:
            raise ValueError("seg or vc")

    plt.savefig(f"{SAVE_PATH}{phase}/{count:03d}.png", dpi=250)
    plt.close()
    count += 1

#-------------------------------------------------------------------------
""" GAN training loop """

def training_loop_GAN(config, model, ds):

    """ config: configuration json 
        model: expects BaseGAN or subclasses (keras.Model)
        ds: Tensorflow dataset """

    EPOCHS = config["EXPT"]["EPOCHS"]
    SAVE_PATH = config["EXPT"]["SAVE_PATH"]
    SAVE_EVERY = config["EXPT"]["SAVE_EVERY"]

    if not os.path.exists(f"{SAVE_PATH}images/GAN/{config['EXPT']['EXPT_NAME']}"):
        os.mkdir(f"{SAVE_PATH}images/GAN/{config['EXPT']['EXPT_NAME']}")
    if not os.path.exists(f"{SAVE_PATH}models/GAN/{config['EXPT']['EXPT_NAME']}"):
        os.mkdir(f"{SAVE_PATH}models/GAN/{config['EXPT']['EXPT_NAME']}")
    if not os.path.exists(f"{SAVE_PATH}logs/GAN/{config['EXPT']['EXPT_NAME']}"):
        os.mkdir(f"{SAVE_PATH}logs/GAN/{config['EXPT']['EXPT_NAME']}")

    # if len(model.generator_metrics.keys()) == 2:
    #     results = {"Discriminator_G": {}, "Discriminator_F": {}}
    # else:
    #     results = {"Discriminator": {}}

    results = {}
    results["g_metric"] = []
    results["d_metric"] = []
    results["train_L1"] = {"global": [], "focal": []}
    results["val_L1"] = {"global": [], "focal": []}
    results["config"] = config
    results["epochs"] = []
    results["time"] = 0

    ds_train, ds_val = ds
    start_time = time.time()
    best_L1 = 1e6

    for epoch in range(EPOCHS):
        results["epochs"].append(epoch + 1)

        # for key, value in model.generator_metrics.items():
        model.generator_metric.reset_states()
        model.train_L1_metric.reset_states()

        # for value in model.discriminator_metrics.values():
        model.discriminator_metric.reset_states()

        for data in ds_train:
            NCE, ACE, mask, coords = data
            model.train_step(NCE, ACE, mask, coords)

        # for key, value in model.generator_metrics.items():
        results["g_metric"].append(float(model.generator_metric.result()))
        results["train_L1"]["global"].append(float(model.train_L1_metric.result()[0]))
        results["train_L1"]["focal"].append(float(model.train_L1_metric.result()[1]))
        
        # for key, value in model.discriminator_metrics.items():
        results["d_metric"].append(float(model.discriminator_metric.result()))

        # for key in model.discriminator_metrics.keys():
        print(f"Train epoch {epoch + 1}, G: {model.generator_metric.result():.4f} D: {model.discriminator_metric.result():.4f}, L1 [global focal]: {model.train_L1_metric.result()}")

        if config["EXPT"]["CV_FOLDS"] != 0:
            model.val_L1_metric.reset_states()

            for data in ds_val:
                NCE, ACE, mask, coords = data
                model.val_step(NCE, ACE, mask, coords)
            
            results["val_L1"]["global"].append(float(model.val_L1_metric.result()[0]))
            results["val_L1"]["focal"].append(float(model.val_L1_metric.result()[1]))

            print(f"Val epoch {epoch + 1}, L1 [global focal]: {model.val_L1_metric.result()}")

        if model.val_L1_metric.result()[1] < best_L1:
            model.save_weights(f"{SAVE_PATH}models/GAN/{config['EXPT']['EXPT_NAME']}/{config['EXPT']['EXPT_NAME']}")
            best_L1 = model.val_L1_metric.result()[1]
            
            with open(f"{SAVE_PATH}logs/GAN/{config['EXPT']['EXPT_NAME']}/best_results.json", 'w') as outfile:
                json.dump(results, outfile, indent=4)

        # TODO: RANDOM EXAMPLE IMAGES
        if (epoch + 1) % SAVE_EVERY == 0:

            for data in ds_val:

                plot_num = 3 if config["EXPT"]["CROP"] else 1
                fig, axs = plt.subplots(plot_num, 4)

                for i in range(plot_num):
                    NCE, ACE, seg, coords = data
                    NCE, ACE, seg = model.crop_ROI(NCE, ACE, seg, coords[:, i, :])
                    NCE, ACE, seg = NCE.numpy(), ACE.numpy(), seg.numpy()
                    pred = model.Generator(NCE).numpy()

                    axs[i, 0].imshow(NCE[0, :, :, 11, 0].T, cmap="gray")
                    axs[i, 0].axis("off")
                    axs[i, 1].imshow(ACE[0, :, :, 11, 0].T, cmap="gray")
                    axs[i, 1].axis("off")
                    axs[i, 2].imshow(pred[0, :, :, 11, 0].T, cmap="gray")
                    axs[i, 2].axis("off")
                    axs[i, 3].imshow(pred[0, :, :, 11, 0].T - ACE[0, :, :, 11, 0].T, cmap="gray")
                    axs[i, 3].axis("off")

                plt.savefig(f"{SAVE_PATH}images/GAN/{config['EXPT']['EXPT_NAME']}/{epoch + 1}.png", dpi=250)
                plt.close()

                break

    results["time"] = (time.time() - start_time) / 3600

    return results