import glob
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os

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

            if (seg == 4).sum() > 0:
                AP["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())
            else:
                print(pred)

            img, _ = nrrd.read(f"{real_path}/Images/{pred[0:6]}HQ{pred[8:11]}.nrrd")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{pred[0:6]}HQ{pred[8:11]}-label.nrrd")

            HQ["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
            HQ["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
            HQ["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())
        
            if (seg == 4).sum() > 0:
                HQ["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())
            else:
                print(pred)

            candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}AC*.nrrd")
            assert len(candidate_imgs) == 1
            img_name = candidate_imgs[0].split('\\')[-1]
            img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

            AC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
            AC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
            AC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            if (seg == 4).sum() > 0:
                AC["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())
            else:
                print(pred)
        
        elif 'VP' in pred:
            VP["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
            VP["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
            VP["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            if (seg == 4).sum() > 0:
                VP["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())
            else:
                print(pred)

            candidate_imgs = glob.glob(f"{real_path}/Images/{pred[0:6]}VC*.nrrd")
            assert len(candidate_imgs) == 1
            img_name = candidate_imgs[0].split('\\')[-1]

            img, _ = nrrd.read(f"{real_path}/Images/{img_name}")
            seg, _ = nrrd.read(f"{real_path}/Segmentations/{img_name[:-5]}-label.nrrd")

            VC["Ao"].append((img * (seg == 1)).sum() / (seg == 1).sum())
            VC["Co"].append((img * (seg == 2)).sum() / (seg == 2).sum())
            VC["Md"].append((img * (seg == 3)).sum() / (seg == 3).sum())

            if (seg == 4).sum() > 0:
                VC["Tu"].append((img * (seg == 4)).sum() / (seg == 4).sum())
            else:
                print(pred)

        else:
            raise ValueError

    ROI_dict = {"Ao": "Aorta", "Co": "Cortex", "Md": "Medulla", "Tu": "Tumour"}

    for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
        print(ROI_dict[ROI])

        for k, d in zip(["HQ", "AC", "AP", "VC", "VP"], [HQ, AC, AP, VC, VP]):
            print(f"{k} {np.median(d[ROI]):.4f} {np.quantile(d[ROI], [0.05, 0.95])}, {len(d[ROI])} samples")

        print("\n")

    # fig, axs = plt.subplots(2, 2)

    # for i, ROI in enumerate(["Ao", "Co", "Md", "Tu"]):
    #     print(ROI_dict[ROI])
    #     print([np.quantile(d[ROI], [0.05, 0.5, 0.95]) for d in [HQ, AC, AP, VC, VP]])
    #     axs.ravel()[i].boxplot([d[ROI] for d in [HQ, AC, AP, VC, VP]])
    #     axs.ravel()[i].set_xticklabels(["NCE", "AC", "ACpred", "VC", "VCpred"])
    #     axs.ravel()[i].set_ylabel("HU")
    #     axs.ravel()[i].set_title(ROI_dict[ROI])

    # plt.show()


#-------------------------------------------------------------------------

if __name__ == "__main__":
    model = "H2_save300_patch"

    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"
    pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/{model}"

    print(model)
    calc_contrast(real_path, pred_path)
