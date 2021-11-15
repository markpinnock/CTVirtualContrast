import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os


real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase3/output/Real"
pred_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase3/output/2_save230"

preds = os.listdir(f"{pred_path}/Segmentations")
subjects = []
HQ = {"Ao": [], "Co": [], "Md": [], "Tu": []}
AC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
VC = {"Ao": [], "Co": [], "Md": [], "Tu": []}
AP = {"Ao": [], "Co": [], "Md": [], "Tu": []}
VP = {"Ao": [], "Co": [], "Md": [], "Tu": []}

for pred in preds:
    img, _ = nrrd.read(f"{pred_path}/Images/{pred.split('-')[0]}.nrrd")
    seg, _ = nrrd.read(f"{pred_path}/Segmentations/{pred}")

    if 'AP' in pred:
        AP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        AP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
        AP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())
        AP["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())

        img, _ = nrrd.read(f"{real_path}/Images/{pred[0:6]}HQ{pred[8:11]}.nrrd")
        seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

        HQ["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        HQ["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
        HQ["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())
        HQ["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())

        candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}AC*.nrrd")
        assert len(candidate_imgs) == 1
        img_name = candidate_imgs[0].split('\\')[-1]
        img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
        seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

        AC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        AC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
        AC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())
        AC["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())
    
    elif 'VP' in pred:
        VP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        VP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
        VP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())
        VP["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())

        candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}VC*.nrrd")
        assert len(candidate_imgs) == 1
        img_name = candidate_imgs[0].split('\\')[-1]

        img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
        seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

        VC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
        VC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
        VC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())
        VC["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())

    else:
        raise ValueError

ROI_dict = {"Ao": "Aorta", "Co": "Cortex", "Md": "Medulla", "Tu": "Tumour"}

fig, axs = plt.subplots(2, 2)

for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
    print(ROI_dict[ROI])
    print([np.quantile(d[ROI], [0.05, 0.5, 0.95]) for d in [HQ, AC, AP, VC, VP]])
    axs.ravel()[i].boxplot([d[ROI] for d in [HQ, AC, AP, VC, VP]])
    axs.ravel()[i].set_xticklabels(["NCE", "AC", "ACpred", "VC", "VCpred"])
    axs.ravel()[i].set_ylabel("HU")
    axs.ravel()[i].set_title(ROI_dict[ROI])

plt.show()
