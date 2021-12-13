import glob
import nrrd
import numpy as np
import os
import skimage.metrics

np.set_printoptions(4)


#-------------------------------------------------------------------------

def calc_metrics(real_path, pred_path):
    pred_imgs = os.listdir(pred_path)
    subjects = []
    MSE = {"AP": [], "VP": [], "HQAC": [], "HQVC": []}
    pSNR = {"AP": [], "VP": [], "HQAC": [], "HQVC": []}
    SSIM = {"AP": [], "VP": [], "HQAC": [], "HQVC": []}
    
    for img in pred_imgs:
        if img[0:6] not in subjects:
            subjects.append(img[0:6])

    for subject in subjects:
        HQ, _ = nrrd.read(glob.glob(f"{real_path}/{subject}HQ*")[0])
        AC, _ = nrrd.read(glob.glob(f"{real_path}/{subject}AC*")[0])
        VC, _ = nrrd.read(glob.glob(f"{real_path}/{subject}VC*")[0])
        AP, _ = nrrd.read(glob.glob(f"{pred_path}/{subject}AP*")[0])
        VP, _ = nrrd.read(glob.glob(f"{pred_path}/{subject}VP*")[0])
    
        MSE["AP"].append(skimage.metrics.mean_squared_error(AC, AP))
        MSE["VP"].append(skimage.metrics.mean_squared_error(VC, VP))
        MSE["HQAC"].append(skimage.metrics.mean_squared_error(AC, HQ))
        MSE["HQVC"].append(skimage.metrics.mean_squared_error(VC, HQ))
        pSNR["AP"].append(skimage.metrics.peak_signal_noise_ratio(AC, AP))
        pSNR["VP"].append(skimage.metrics.peak_signal_noise_ratio(VC, VP))
        pSNR["HQAC"].append(skimage.metrics.peak_signal_noise_ratio(AC, HQ))
        pSNR["HQVC"].append(skimage.metrics.peak_signal_noise_ratio(AC, AP))
        SSIM["AP"].append(skimage.metrics.structural_similarity(AC, AP))
        SSIM["VP"].append(skimage.metrics.structural_similarity(VC, VP))
        SSIM["HQAC"].append(skimage.metrics.structural_similarity(AC, HQ))
        SSIM["HQVC"].append(skimage.metrics.structural_similarity(VC, HQ))

        #print(subject)

        #for key in ["AP", "HQAC", "VP", "HQVC"]:
        #    print(f"{key} {MSE[key][-1]:.2f}, {pSNR[key][-1]:.2f}, {SSIM[key][-1]:.4f}")

        #print("\n")

    print("Overall")

    for key in ["AP", "VP"]: #["AP", "HQAC", "VP", "HQVC"]:
        print(f"{key} {np.median(MSE[key]):.4f} {np.quantile(MSE[key], [0.05, 0.95])}, {np.median(pSNR[key]):.4f} {np.quantile(pSNR[key], [0.05, 0.95])}, {np.median(SSIM[key]):.4f} {np.quantile(SSIM[key], [0.05, 0.95])}")


#-------------------------------------------------------------------------

if __name__ == "__main__":

    model = "UNetT_save1000"
    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real/Images"
    pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/{model}/Images"

    print(model)
    calc_metrics(real_path, pred_path)
