import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time


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

    x = np.zeros([1, 512 // config["EXPT"]["DOWN_SAMP"], 512 // config["EXPT"]["DOWN_SAMP"], 12, in_ch], dtype=np.float32)
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


def training_loop_UNet(config, phase, model, ds, show):
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
    EPOCHS = config["EPOCHS"]
    SAVE_PATH = config["SAVE_PATH"]

    if not os.path.exists(f"{SAVE_PATH}images/GAN/{config['EXPT_NAME']}"):
        os.mkdir(f"{SAVE_PATH}images/GAN/{config['EXPT_NAME']}")
    if not os.path.exists(f"{SAVE_PATH}models/GAN/{config['EXPT_NAME']}"):
        os.mkdir(f"{SAVE_PATH}models/GAN/{config['EXPT_NAME']}")
    if not os.path.exists(f"{SAVE_PATH}logs/GAN/{config['EXPT_NAME']}"):
        os.mkdir(f"{SAVE_PATH}logs/GAN/{config['EXPT_NAME']}")

    if len(model.generator_metrics.keys()) == 2:
        results = {"Discriminator_G": {}, "Discriminator_F": {}}
    else:
        results = {"Discriminator": {}}

    for val in results.values():
        val["losses"] = {"G": [], "D1": [], "D2": []}
        val["train_metrics"] = {"global": [], "focal": []}
    
    results["val_metrics"] = {"global": [], "focal": []}
    results["config"] = config
    results["epochs"] = []
    results["time"] = 0

    ds_train, ds_val = ds
    start_time = time.time()
    best_L1 = 1e6

    for epoch in range(EPOCHS):
        results["epochs"].append(epoch + 1)

        for key, value in model.generator_metrics.items():
            value["g_metric"].reset_states()
            value["g_L1"].reset_states()

        for value in model.discriminator_metrics.values():
            value["d_metric_1"].reset_states()
            value["d_metric_2"].reset_states()

        for data in ds_train:
            NCE, ACE, mask, coords = data
            model.ROI_train_step(NCE, ACE, mask, coords)

        for key, value in model.generator_metrics.items():
            results[key]["losses"]["G"].append(float(value['g_metric'].result()))
            results[key]["train_metrics"]["global"].append(float(value["g_L1"].result()[0]))
            results[key]["train_metrics"]["focal"].append(float(value["g_L1"].result()[1]))
        
        for key, value in model.discriminator_metrics.items():
            results[key]["losses"]["D1"].append(float(value['d_metric_1'].result()))
            results[key]["losses"]["D2"].append(float(value['d_metric_2'].result()))

        for key in model.discriminator_metrics.keys():
            print(f"Train epoch {epoch + 1} {key}, G: {model.generator_metrics[key]['g_metric'].result():.4f} D1: {model.discriminator_metrics[key]['d_metric_1'].result():.4f}, D2: {model.discriminator_metrics[key]['d_metric_2'].result():.4f}, L1 [global focal]: {model.generator_metrics[key]['g_L1'].result()}")
        
        if config["CV_FOLDS"] != 0:
            model.generator_val_metric.reset_states()

            for data in ds_val:
                NCE, ACE, mask, coords = data
                model.val_step(NCE, ACE, mask, coords)
            
            results["val_metrics"]["global"].append(float(model.generator_val_metric.result()[0]))
            results["val_metrics"]["focal"].append(float(model.generator_val_metric.result()[1]))

            print(f"Val epoch {epoch + 1}, L1 [global focal]: {model.generator_val_metric.result()}")

        if model.generator_val_metric.result()[1] < best_L1 and epoch > 75:
            model.save_weights(f"{SAVE_PATH}models/GAN/{config['EXPT_NAME']}/{config['EXPT_NAME']}")
            best_L1 = model.generator_val_metric.result()[1]
            
            with open(f"{SAVE_PATH}logs/GAN/{config['EXPT_NAME']}/best_results.json", 'w') as outfile:
                json.dump(results, outfile, indent=4)

        # TODO: RANDOM EXAMPLE IMAGES
        if (epoch >= 70 and epoch % 10 == 0) or config["VERBOSE"]:
            for data in ds_val:
                NCE, ACE, seg, coords = data
                # TODO: for loop
                NCE, ACE, seg = model.crop_ROI(NCE, ACE, seg, coords[:, 1, :])
                pred = model.Generator(NCE).numpy()

                fig, axs = plt.subplots(2, 3)
                axs[0, 0].imshow(np.flipud(NCE[0, :, :, 11, 0]), cmap='gray', origin='lower')
                axs[0, 0].axis("off")
                axs[0, 1].imshow(np.flipud(ACE[0, :, :, 11, 0]), cmap='gray', origin='lower')
                axs[0, 1].axis("off")
                axs[0, 2].imshow(np.flipud(pred[0, :, :, 11, 0]), cmap='gray', origin='lower')
                axs[0, 2].axis("off")
                axs[1, 0].imshow(np.flipud(seg[0, :, :, 11, 0]), cmap='gray', origin='lower')
                axs[1, 0].axis("off")
                axs[1, 1].imshow(np.flipud(pred[0, :, :, 11, 0] + NCE[0, :, :, 11, 0] - ACE[0, :, :, 11, 0]), cmap='gray', origin='lower')
                axs[1, 1].axis("off")
                axs[1, 2].imshow(np.flipud(pred[0, :, :, 11, 0] - ACE[0, :, :, 11, 0]), cmap='gray', origin='lower')
                axs[1, 2].axis("off")

                if show:
                    plt.show()
                else:
                    plt.savefig(f"{SAVE_PATH}images/GAN/{config['EXPT_NAME']}/{epoch + 1}.png", dpi=250)
                    plt.close()
                
                break

    results["time"] = (time.time() - start_time) / 3600

    return results