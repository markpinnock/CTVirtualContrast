import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import scipy.stats as stat

from .util import bootstrap

np.set_printoptions(4)

#-------------------------------------------------------------------------

def calc_contrast(real_path, pred_path, slicewise=False):
    preds = os.listdir(f"{pred_path}/Images")
    HQ = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    AC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    VC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    AP = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    VP = {"Ao": [], "Co": [], "Md": [], "Tu": []}
    epsilon = 1e-8

    for pred in preds:
        img, _ = nrrd.read(f"{pred_path}/Images/{pred}")
        seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

        if 'AP' in pred:
            if slicewise:
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                AP["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                AP["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                AP["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))

            else:
                AP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                AP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                AP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            try:
                seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
            except:
                print(pred)
            else:
                AP["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            img, _ = nrrd.read(f"{real_path}/Images/{pred[0:6]}HQ{pred[8:11]}.nrrd")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

            if slicewise:
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                HQ["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                HQ["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                HQ["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))

            else:
                HQ["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                HQ["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                HQ["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            try:
                seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
            except:
                print(pred)
            else:
                HQ["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}AC*.nrrd")
            assert len(candidate_imgs) == 1
            img_name = candidate_imgs[0].split('\\')[-1]
            img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

            if slicewise:
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                AC["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                AC["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                AC["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))

            else:
                AC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                AC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                AC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            try:
                seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{img_name[:-5]}-label.nrrd")
            except:
                print(pred)
            else:
                AC["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        
        elif 'VP' in pred:
            if slicewise:
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                VP["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                VP["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                VP["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))

            else:
                VP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                VP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                VP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            try:
                seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")
            except:
                print(pred)
            else:
                VP["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

            candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}VC*.nrrd")
            assert len(candidate_imgs) == 1
            img_name = candidate_imgs[0].split('\\')[-1]

            img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

            if slicewise:
                slices = np.unique(np.argwhere((seg > 0) & (seg < 4))[:, 2])
                VC["Ao"].append((img[:, :, slices] * (seg[:, :, slices] == 1)).sum(axis=(0, 1)) / (seg[:, :, slices] == 1).sum(axis=(0, 1)))
                VC["Co"].append((img[:, :, slices] * (seg[:, :, slices] == 2)).sum(axis=(0, 1)) / (seg[:, :, slices] == 2).sum(axis=(0, 1)))
                VC["Md"].append((img[:, :, slices] * (seg[:, :, slices] == 3)).sum(axis=(0, 1)) / (seg[:, :, slices] == 3).sum(axis=(0, 1)))

            else:
                VC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
                VC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
                VC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            try:
                seg, _ = nrrd.read(f"{real_path}/Segmentations_Tumour/{img_name[:-5]}-label.nrrd")
            except:
                print(pred)
            else:
                VC["Tu"].append((img * (seg == 1)).sum() / (seg == 1).sum())

        else:
            raise ValueError

    return {"HQ": HQ, "AC": AC, "AP": AP, "VC": VC, "VP": VP}


#-------------------------------------------------------------------------

def get_regression(m, d):
    result = stat.linregress(m, d)
    std_res = np.std(d - (m * result.slope + result.intercept))

    return result.slope, result.intercept, std_res, result.pvalue


#-------------------------------------------------------------------------

def display_bland_altman(expt1, expt2, results):

    fig, axs = plt.subplots(2, 2)

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        diff = np.array(results[expt1][ROI]) - np.array(results[expt2][ROI])
        mean_ = (np.array(results[expt1][ROI]) + np.array(results[expt2][ROI])) / 2
        slope, intercept, std_res, p = get_regression(mean_, diff)
        fitted = mean_ * slope + intercept
        axs.ravel()[i].scatter(mean_, diff, s=80, c='k', marker='+')

        if p * 48 > 0.05:
            result = stat.ttest_1samp(diff, 0)
            print(f"{expt1}, {expt2}, {ROI}, bias {np.mean(diff):.0f}, LoA {np.round(1.96 * np.std(diff))}, p-value {p}, t-test {(result.statistic, result.pvalue)}")
            axs.ravel()[i].axhline(np.mean(diff), c='k', ls='-', label="Bias")
            axs.ravel()[i].axhline(np.mean(diff) - 1.96 * np.std(diff), c='r', ls='-')
            axs.ravel()[i].axhline(np.mean(diff) + 1.96 * np.std(diff), c='r', ls='-', label="95% LoA")

        else:
            print(f"{expt1}, {expt2}, {ROI}, intercept {np.round(intercept)}, slope {np.round(slope, 2)}, LoA {np.round(1.96 * std_res)}, p-value {p}")
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

def rep_coeff(expt1, expt2, results):
    for ROI in ["Ao", "Co", "Md", "Tu"]:
        diff = np.vstack(results[expt1][ROI]) - np.vstack(results[expt2][ROI])
        between_group = diff.shape[1] * np.sum(np.square(np.mean(diff, axis=1) - np.mean(diff))) / diff.shape[0]
        within_group = np.sum(np.square(diff - np.mean(diff, axis=1, keepdims=True))) / (diff.shape[0] * diff.shape[1] - diff.shape[1])
        print(f"{expt1}, {expt2}, {ROI}, between {between_group} within {within_group}")


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

if __name__ == "__main__":
    model = "H2_save300_patch"

    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"
    pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/{model}"

    print(model)
    contrasts = calc_contrast(real_path, pred_path, slicewise=False)
    ROI_dict = {"Ao": "Aorta", "Co": "Cortex", "Md": "Medulla", "Tu": "Tumour"}

    display_bland_altman("AC", "AP", contrasts)
    display_bland_altman("VC", "VP", contrasts)
    
    contrasts = calc_contrast(real_path, pred_path, slicewise=True)
    rep_coeff("AC", "AP", contrasts) # stat.ttest_1samp
    exit()
    # print(bootstrap_and_display("HQ", None, "Ao", contrasts))
    # print(bootstrap_and_display("HQ", None, "Co", contrasts))
    # print(bootstrap_and_display("HQ", None, "Md", contrasts))
    # print(bootstrap_and_display("HQ", None, "Tu", contrasts))
    # print(bootstrap_and_display("AC", None, "Ao", contrasts))
    # print(bootstrap_and_display("AC", None, "Co", contrasts))
    # print(bootstrap_and_display("AC", None, "Md", contrasts))
    # print(bootstrap_and_display("AC", None, "Tu", contrasts))
    # print(bootstrap_and_display("VC", None, "Ao", contrasts))
    # print(bootstrap_and_display("VC", None, "Co", contrasts))
    # print(bootstrap_and_display("VC", None, "Md", contrasts))
    # print(bootstrap_and_display("VC", None, "Tu", contrasts))

    print(bootstrap_and_display("AP", None, "Ao", contrasts))
    print(bootstrap_and_display("AP", None, "Co", contrasts))
    print(bootstrap_and_display("AP", None, "Md", contrasts))
    print(bootstrap_and_display("AP", None, "Tu", contrasts))
    print(bootstrap_and_display("VP", None, "Ao", contrasts))
    print(bootstrap_and_display("VP", None, "Co", contrasts))
    print(bootstrap_and_display("VP", None, "Md", contrasts))
    print(bootstrap_and_display("VP", None, "Tu", contrasts))
    print(bootstrap_and_display("AC", "AP", "Ao", contrasts))
    print(bootstrap_and_display("AC", "AP", "Co", contrasts))
    print(bootstrap_and_display("AC", "AP", "Md", contrasts))
    print(bootstrap_and_display("AC", "AP", "Tu", contrasts))
    print(bootstrap_and_display("VC", "VP", "Ao", contrasts))
    print(bootstrap_and_display("VC", "VP", "Co", contrasts))
    print(bootstrap_and_display("VC", "VP", "Md", contrasts))
    print(bootstrap_and_display("VC", "VP", "Tu", contrasts))
