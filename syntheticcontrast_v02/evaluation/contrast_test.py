import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import pandas as pd
import scipy.stats as stat

from .util import bootstrap

np.set_printoptions(4)

#-------------------------------------------------------------------------

def calc_contrast(real_path, pred_path, slicewise=False, save_path=None, model_save_name=None):
    preds = os.listdir(f"{pred_path}/Images")
    HQ = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    AC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    VC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    AP = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    VP = {"Ao": [], "Co": [], "Md": [], "Tu": []}

    for pred in preds:
        img, _ = nrrd.read(f"{pred_path}/Images/{pred}")
        seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

        if 'AP' in pred:
            # Get predicted AC phase
            if slicewise:
                # Get individual slices for masks
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                AP["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                AP["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                AP["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))
                slices = np.unique(np.argwhere(seg == 4)[:, 2])

                # Use slices for tumour mask
                if slices.sum() == 0:
                    print(pred)
                else:
                    AP["Tu"].append((img[:, :, slices] * (seg[:, :, slices] == 4)).sum(axis=(0, 1)) / (seg[:, :, slices] == 4).sum(axis=(0, 1)))

            else:
                # Average slices for masks
                AP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                AP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                AP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
                except:
                    AP["Tu"].append(np.nan)
                    #print(pred)
                else:
                    AP["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            img, _ = nrrd.read(f"{real_path}/Images/{pred[0:6]}HQ{pred[8:11]}.nrrd")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

            # Get ground truth HQ phase
            if slicewise:
                # Get individual slices for masks
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                HQ["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                HQ["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                HQ["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))
                slices = np.unique(np.argwhere(seg == 4)[:, 2])

                # Use slices for tumour mask
                if slices.sum() == 0:
                    print(pred)
                else:
                    HQ["Tu"].append((img[:, :, slices] * (seg[:, :, slices] == 4)).sum(axis=(0, 1)) / (seg[:, :, slices] == 4).sum(axis=(0, 1)))

            else:
                # Average slices for masks
                HQ["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                HQ["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                HQ["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
                except:
                    HQ["Tu"].append(np.nan)
                    #print(pred)
                else:
                    HQ["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}AC*.nrrd")
            assert len(candidate_imgs) == 1
            img_name = candidate_imgs[0].split('\\')[-1]
            img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

            # Get ground truth AC phase
            if slicewise:
                # Get individual slices for masks
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                AC["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                AC["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                AC["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))
                slices = np.unique(np.argwhere(seg == 4)[:, 2])

                # Use slices for tumour mask
                if slices.sum() == 0:
                    print(pred)
                else:
                    AC["Tu"].append((img[:, :, slices] * (seg[:, :, slices] == 4)).sum(axis=(0, 1)) / (seg[:, :, slices] == 4).sum(axis=(0, 1)))

            else:
                # Average slices for masks
                AC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                AC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                AC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{img_name[:-5]}-label.nrrd")
                except:
                    AC["Tu"].append(np.nan)
                    #print(pred)
                else:
                    AC["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        
        elif 'VP' in pred:
            # Get predicted VC phase
            if slicewise:
                # Get individual slices for masks
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                VP["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                VP["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                VP["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))
                slices = np.unique(np.argwhere(seg == 4)[:, 2])

                # Use slices for tumour mask
                if slices.sum() == 0:
                    print(pred)
                else:
                    VP["Tu"].append((img[:, :, slices] * (seg[:, :, slices] == 4)).sum(axis=(0, 1)) / (seg[:, :, slices] == 4).sum(axis=(0, 1)))

            else:
                # Average slices for masks
                VP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                VP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                VP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
                except:
                    VP["Tu"].append(np.nan)
                    #print(pred)
                else:
                    VP["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}VC*.nrrd")
            assert len(candidate_imgs) == 1
            img_name = candidate_imgs[0].split('\\')[-1]

            img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

            # Get ground truth VC phase
            if slicewise:
                # Get individual slices for masks
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                VC["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                VC["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                VC["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))
                slices = np.unique(np.argwhere(seg == 4)[:, 2])

                # Use slices for tumour mask
                if slices.sum() == 0:
                    print(pred)
                else:
                    VC["Tu"].append((img[:, :, slices] * (seg[:, :, slices] == 4)).sum(axis=(0, 1)) / (seg[:, :, slices] == 4).sum(axis=(0, 1)))

            else:
                # Average slices for masks
                VC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                VC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                VC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

                # Average whole tumour mask
                try:
                    seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{img_name[:-5]}-label.nrrd")
                except:
                    VC["Tu"].append(np.nan)
                    #print(pred)
                else:
                    VC["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

        else:
            raise ValueError
        
    if save_path is not None:
        gt_cols = pd.MultiIndex.from_product([["NCE", "CME", "NGE"], ["Aorta", "Cortex", "Medulla", "Tumour"]])
        pred_cols = pd.MultiIndex.from_product([["CME", "NGE"], ["Aorta", "Cortex", "Medulla", "Tumour"]])
        gt_df = pd.DataFrame(columns=gt_cols)
        pred_df = pd.DataFrame(columns=pred_cols)

        for ROI_old, ROI_new in zip(["Ao", "Co", "Md", "Tu"], ["Aorta", "Cortex", "Medulla", "Tumour"]):
            gt_df["NCE", ROI_new] = HQ[ROI_old]
            gt_df["CME", ROI_new] = AC[ROI_old]
            gt_df["NGE", ROI_new] = VC[ROI_old]

        for ROI_old, ROI_new in zip(["Ao", "Co", "Md", "Tu"], ["Aorta", "Cortex", "Medulla", "Tumour"]):
            pred_df["CME", ROI_new] = AP[ROI_old]
            pred_df["NGE", ROI_new] = VP[ROI_old]

        print(pred_df)
        gt_df.to_csv(f"{save_path}/contrast_gt.csv")
        pred_df.to_csv(f"{save_path}/contrast_{model_save_name}.csv")

    return {"HQ": HQ, "AC": AC, "AP": AP, "VC": VC, "VP": VP}


#-------------------------------------------------------------------------

def get_regression(m, d):
    result = stat.linregress(m, d)
    std_res = np.std(d - (m * result.slope + result.intercept))

    return result.slope, result.intercept, std_res, result.pvalue


#-------------------------------------------------------------------------

def stats_bland_altman(expt1, expt2, results):

    fig, axs = plt.subplots(2, 2)

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        diff = np.array(results[expt1][ROI]) - np.array(results[expt2][ROI])
        mean_ = (np.array(results[expt1][ROI]) + np.array(results[expt2][ROI])) / 2
        slope, intercept, std_res, p = get_regression(mean_, diff)
        fitted = mean_ * slope + intercept
        axs.ravel()[i].scatter(mean_, diff, s=80, c='k', marker='+')

        print(f"{expt1}, {expt2}, {ROI}, intercept {np.round(intercept)}, slope {np.round(slope, 2)}, mean bias {np.round(diff.mean())}, LoA {np.round(1.96 * std_res)}, p-value {p}")
        result = stat.ttest_1samp(diff, 0)
        print(f"{expt1}, {expt2}, {ROI}, bias {np.mean(diff):.0f}, LoA {np.round(1.96 * np.std(diff))}, t-test {(result.statistic, result.pvalue)}")

        axs.ravel()[i].plot(mean_, fitted, c='k', ls='-', label="Bias")
        axs.ravel()[i].plot(mean_, fitted - 1.96 * std_res, c='r', ls='-')
        axs.ravel()[i].plot(mean_, fitted + 1.96 * std_res, c='r', ls='-', label="95% LoA")

        axs.ravel()[i].axhline(0.0, c='k', ls='--')
        axs.ravel()[i].set_xlabel(r"$(HU_{actual} + HU_{pred}) / 2$")
        axs.ravel()[i].set_ylabel(r"$HU_{actual} - HU_{pred}$")
        axs.ravel()[i].set_title(ROI_dict[ROI])
        axs.ravel()[i].legend()

    plt.show()


#-------------------------------------------------------------------------

def display_bland_altman(gt_phase, expts):
    pred_phase = f"{gt_phase[0]}P"
    colour_dict = dict(zip(expts.keys(), ['k', 'r', 'b', 'g']))

    fig, axs = plt.subplots(2, 2)

    pooled_x = {}

    for ROI in ["Ao", "Co", "Md", "Tu"]:
        x = []

        for expt in colour_dict.keys():
            x.append((np.array(expts[expt][gt_phase][ROI]) + np.array(expts[expt][pred_phase][ROI])) / 2)

        pooled_x[ROI] = np.hstack(x)

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        for expt in colour_dict.keys():
            diff = np.array(expts[expt][gt_phase][ROI]) - np.array(expts[expt][pred_phase][ROI])
            mean_ = (np.array(expts[expt][gt_phase][ROI]) + np.array(expts[expt][pred_phase][ROI])) / 2
            slope, intercept, std_res, p = get_regression(mean_, diff)
            fitted = mean_ * slope + intercept
            print(f"{expt} adjusted mean {pooled_x[ROI].mean() * slope + intercept}, LoA {1.96 * std_res}")
            axs.ravel()[i].plot(mean_, fitted, c=colour_dict[expt], ls='-', label=expt)
            axs.ravel()[i].plot(mean_, fitted - 1.96 * std_res, c=colour_dict[expt], ls='--')
            axs.ravel()[i].plot(mean_, fitted + 1.96 * std_res, c=colour_dict[expt], ls='--')

        axs.ravel()[i].axhline(0.0, c='k', ls='-')
        axs.ravel()[i].set_xlabel(r"$(HU_{actual} + HU_{pred}) / 2$")
        axs.ravel()[i].set_ylabel(r"$HU_{actual} - HU_{pred}$")
        axs.ravel()[i].set_title(ROI_dict[ROI])
        axs.ravel()[i].legend()

    plt.show()


#-------------------------------------------------------------------------

def get_adj_means(gt_phase, expts):
    pred_phase = f"{gt_phase[0]}P"
    pooled_x = {}
    pooled_y = {}

    for ROI in ["Ao", "Co", "Md", "Tu"]:
        x = []
        y = []

        for expt in expts.keys():
            x.append((np.array(expts[expt][gt_phase][ROI]) + np.array(expts[expt][pred_phase][ROI])) / 2)
            y.append(np.array(expts[expt][gt_phase][ROI]) - np.array(expts[expt][pred_phase][ROI]))

        pooled_x[ROI] = np.hstack(x)

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        print(ROI)
        com_m, _, _, _ = get_regression(pooled_x[ROI], pooled_y[ROI])
        grand_mean = pooled_x[ROI].mean()

        for expt in expts.keys():
            y = np.array(expts[expt][gt_phase][ROI]) - np.array(expts[expt][pred_phase][ROI])
            x = (np.array(expts[expt][gt_phase][ROI]) + np.array(expts[expt][pred_phase][ROI])) / 2
            _, _, std_res, _ = get_regression(x, y)
            print(f"{expt} adjusted mean {np.round(np.mean(y - (x - grand_mean) * com_m))}, LoA {np.round(1.96 * std_res)}")


#-------------------------------------------------------------------------

def rep_coeff(expt1, expt2, results):
    for ROI in ["Ao", "Co", "Md", "Tu"]:
        diff = np.vstack(results[expt1][ROI]) - np.vstack(results[expt2][ROI])
        between_group = diff.shape[1] * np.sum(np.square(np.mean(diff, axis=1) - np.mean(diff))) / diff.shape[0]
        within_group = np.sum(np.square(diff - np.mean(diff, axis=1, keepdims=True))) / (diff.shape[0] * diff.shape[1] - diff.shape[1])
        print(f"{expt1}, {expt2}, {ROI}, between {between_group}, within {within_group}, repeatability {np.sqrt(within_group) * 1.96 * np.sqrt(2)}")


#-------------------------------------------------------------------------

def display_boxplot(expts):

    fig, axs = plt.subplots(2, 2)

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        print(ROI_dict[ROI])
        print([np.quantile(d[ROI], [0.05, 0.5, 0.95]) for d in expts.values()])
        axs.ravel()[i].boxplot([d[ROI] for d in expts.values()])
        axs.ravel()[i].set_xticklabels(["NCE", "AC", "ACpred", "VC", "VCpred"])
        axs.ravel()[i].set_ylabel("HU")
        axs.ravel()[i].set_title(ROI_dict[ROI])

    plt.show()


#-------------------------------------------------------------------------

def bootstrap_and_display(expt1, expt2, ROI, results):
    if expt2 is None:
        diff = np.median(results[expt1][ROI])
        boot_results = bootstrap(np.array(results[expt1][ROI]), None, N=100000)
    else:
        diff = np.median(results[expt1][ROI]) - np.median(results[expt2][ROI])
        boot_results = bootstrap(np.array(results[expt1][ROI]), np.array(results[expt2][ROI]), N=100000)

    h = plt.hist(boot_results, bins=20)
    plt.axvline(diff, c='k', ls='--')
    plt.errorbar(x=diff, y=(0.75 * np.max(h[0])), xerr=(1.96 * np.std(boot_results)))
    plt.title(f"{expt1} - {expt2} for {ROI_dict[ROI]}")
    plt.show()

    # Pivot method
    percentiles = np.quantile(boot_results, [0.975, 0.025]) # NB: these are switched

    return expt1, expt2, ROI, diff, 2 * diff - percentiles, f"Bias {np.mean(boot_results) - diff}, std err {np.std(boot_results)}"


#-------------------------------------------------------------------------

def display_bootstraps(contrasts):
    for phase in ["HQ", "AC", "VC", "AP", "VP"]:
        for ROI in ["Ao", "Co", "Md", "Tu"]:
            print(bootstrap_and_display(phase, None, ROI, contrasts))


#-------------------------------------------------------------------------

if __name__ == "__main__":

    ROI_dict = {"Ao": "Aorta", "Co": "Cortex", "Md": "Medulla", "Tu": "Tumour"}
    models = {
        "UNetACVC": "unetbase",
        "UNetT_save1000": "unetphase",
        "CycleGANT_save880": "cyclegan",
        "2_save230": "p2p",
        "2_save170_patch": "p2ppatch",
        "H2_save280": "hyperp2p",
        "H2_save300_patch": "hyperp2ppatch"
    }

    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"
    results = []

    for model, model_save in models.items():
        pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/{model}"
        save_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/results"
        print(model)
        results.append(calc_contrast(real_path, pred_path, slicewise=False, save_path=save_path, model_save_name=model_save))
    exit()
    results_dict = dict(zip(model_names, results))

    #display_bootstraps(results_dict["UNet-Base"])

#    print("UNet-Base")
#    stats_bland_altman("AC", "AP", results_dict["UNet-Base"])
#    print("UNet-Phase")
#    stats_bland_altman("AC", "AP", results_dict["UNet-Phase"])
#    print("Pix2Pix")
#    stats_bland_altman("AC", "AP", results_dict["Pix2Pix"])
#    print("CycleGAN")
#    stats_bland_altman("AC", "AP", results_dict["CycleGAN"])
#    print("UNet-Base")
#    stats_bland_altman("VC", "VP", results_dict["UNet-Base"])
#    print("UNet-Phase")
#    stats_bland_altman("VC", "VP", results_dict["UNet-Phase"])
#    print("Pix2Pix")
#    stats_bland_altman("VC", "VP", results_dict["Pix2Pix"])
#    print("CycleGAN")
#    stats_bland_altman("VC", "VP", results_dict["CycleGAN"])
#    exit()

    get_adj_means("AC", results_dict)
    get_adj_means("VC", results_dict)

#    display_bland_altman("AC", results_dict)
#    display_bland_altman("VC", results_dict)

    #contrasts = calc_contrast(real_path, pred_path, slicewise=True)
    #rep_coeff("AC", "AP", contrasts)
    #rep_coeff("VC", "VP", contrasts)
    exit()