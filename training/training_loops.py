import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time


def training_loop_UNet(config, phase, model, ds, show):
    EPOCHS = config["HYPERPARAMS"]["EPOCHS"]
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


def training_loop_GAN(config, model, ds, show):
    EPOCHS = config["HYPERPARAMS"]["EPOCHS"]
    SAVE_PATH = config["SAVE_PATH"]

    if not os.path.exists(f"{SAVE_PATH}images/"):
        os.mkdir(f"{SAVE_PATH}images/GAN/")
    if not os.path.exists(f"{SAVE_PATH}images/"):
        os.mkdir(f"{SAVE_PATH}images/GAN/")
    if not os.path.exists(f"{SAVE_PATH}models/"):
        os.mkdir(f"{SAVE_PATH}models/")
    if not os.path.exists(f"{SAVE_PATH}models/GAN/"):
        os.mkdir(f"{SAVE_PATH}models/GAN/")
    if not os.path.exists(f"{SAVE_PATH}logs/"):
        os.mkdir(f"{SAVE_PATH}logs/")
    if not os.path.exists(f"{SAVE_PATH}logs/GAN/"):
        os.mkdir(f"{SAVE_PATH}logs/GAN/")

    results = {}
    results["config"] = config
    results["epochs"] = []
    results["losses"] = {"G": [], "D1": [], "D2": []}
    results["train_metrics"] = {"global": [], "focal": []}
    results["val_metrics"] = {"global": [], "focal": []}
    results["time"] = 0

    ds_train, ds_val = ds
    start_time = time.time()
    best_L1 = 1e3

    for epoch in range(EPOCHS):
        results["epochs"].append(epoch + 1)
        model.metric_dict["g_metric"].reset_states()
        model.metric_dict["d_metric_1"].reset_states()
        model.metric_dict["d_metric_2"].reset_states()
        model.L1metric.reset_states()

        for data in ds_train:
            NCE, ACE, mask = data
            model.train_step(NCE, ACE, mask)

        results["losses"]["G"].append(float(model.metric_dict['g_metric'].result()))
        results["losses"]["D1"].append(float(model.metric_dict['d_metric_1'].result()))
        results["losses"]["D2"].append(float(model.metric_dict['d_metric_2'].result()))
        results["train_metrics"]["global"].append(float(model.L1metric.result()[0]))
        results["train_metrics"]["focal"].append(float(model.L1metric.result()[1]))

        print(f"Train epoch {epoch + 1}, G: {model.metric_dict['g_metric'].result():.4f} D1: {model.metric_dict['d_metric_1'].result():.4f}, D2: {model.metric_dict['d_metric_2'].result():.4f}, L1 [global focal]: {model.L1metric.result()}")
        
        if config["CV_FOLDS"] != 0:
            model.L1metric.reset_states()

            for data in ds_val:
                NCE, ACE, mask = data
                model.val_step(NCE, ACE, mask)
            
            results["val_metrics"]["global"].append(float(model.L1metric.result()[0]))
            results["val_metrics"]["focal"].append(float(model.L1metric.result()[1]))

            print(f"Val epoch {epoch + 1}, L1 [global focal]: {model.L1metric.result()}")

            if model.L1metric.result()[1] < best_L1:
                model.save_weights(f"{SAVE_PATH}models/GAN/GAN")
                best_L1 = model.L1metric.result()[1]
                
                with open(f"{SAVE_PATH}logs/GAN/best_results.json", 'w') as outfile:
                    json.dump(results, outfile, indent=4)

        # TODO: RANDOM EXAMPLE IMAGES
        for data in ds_val:
            NCE, ACE, seg = data
            pred = model.Generator(NCE).numpy()

            fig, axs = plt.subplots(2, 3)
            axs[0, 0].imshow(np.flipud(NCE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[0, 0].axis("off")
            axs[0, 1].imshow(np.flipud(ACE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[0, 1].axis("off")
            axs[0, 2].imshow(np.flipud(pred[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[0, 2].axis("off")
            axs[1, 0].imshow(np.flipud(seg[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[1, 0].axis("off")
            axs[1, 1].imshow(np.flipud(pred[3, :, :, 11, 0] + NCE[3, :, :, 11, 0] - ACE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[1, 1].axis("off")
            axs[1, 2].imshow(np.flipud(pred[3, :, :, 11, 0] - ACE[3, :, :, 11, 0]), cmap='gray', origin='lower')
            axs[1, 2].axis("off")

            if show:
                plt.show()
            else:
                plt.savefig(f"{SAVE_PATH}images/GAN/{epoch + 1}.png", dpi=250)
                plt.close()
            
            break

    results["time"] = (time.time() - start_time) / 3600

    return results