import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os

from .util import bootstrap

np.set_printoptions(4)

#-------------------------------------------------------------------------

def calc_contrast(real_path, pred_path):
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

def display_bland_altman(expt1, expt2, results):

    fig, axs = plt.subplots(2, 2)

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        diff = np.array(results[expt1][ROI]) - np.array(results[expt2][ROI])
        mean_ = (np.array(results[expt1][ROI]) + np.array(results[expt2][ROI])) / 2
        axs.ravel()[i].scatter(mean_, diff, c='r', marker='+')
        axs.ravel()[i].axhline(np.mean(diff), c='r', ls='--', label="Mean")
        axs.ravel()[i].axhline(np.mean(diff) - 1.96 * np.std(diff), c='k', ls='--')
        axs.ravel()[i].axhline(np.mean(diff) + 1.96 * np.std(diff), c='k', ls='--', label="95% CI")
        axs.ravel()[i].axhline(0.0, c='k', ls='-')
        axs.ravel()[i].set_xlabel("(HUpred + HUactual) / 2")
        axs.ravel()[i].set_ylabel("HUpred - HUactual")
        axs.ravel()[i].set_title(ROI_dict[ROI])
        axs.ravel()[i].legend()

    plt.show()


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

    return expt1, expt2, ROI, diff, np.quantile(boot_results, [0.025, 0.975])


#-------------------------------------------------------------------------

if __name__ == "__main__":
    model = "H2_save300_patch"

    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"
    pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/{model}"

    print(model)
    contrasts = calc_contrast(real_path, pred_path)

    ROI_dict = {"Ao": "Aorta", "Co": "Cortex", "Md": "Medulla", "Tu": "Tumour"}

    for ROI in ["Ao", "Co", "Md", "Tu"]:
        print(ROI_dict[ROI])

        for k, d in contrasts.items():
            print(f"{k} {np.median(d[ROI]):.4f} {np.quantile(d[ROI], [0.05, 0.95])}, {len(d[ROI])} samples")

        print("\n")

    display_bland_altman("AC", "AP", contrasts)

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
