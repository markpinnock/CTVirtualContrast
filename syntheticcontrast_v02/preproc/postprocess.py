import glob
import numpy as np
import os
import SimpleITK as itk


def NRRDConv(image_path, pred_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    preds = [name for name in os.listdir(pred_path)]

    for pred_name in preds:
        HQ_name = f"{pred_name[0:6]}HQ{pred_name[8:]}"

        if "AP" in pred_name:
            CE_candidates = glob.glob(f"{image_path}/{pred_name[0:6]}AC*")
        elif "VP" in pred_name:
            CE_candidates = glob.glob(f"{image_path}/{pred_name[0:6]}VC*")
        else:
            raise ValueError

        assert len(CE_candidates) == 1, CE_candidates

        pred_nrrd = itk.GetImageFromArray(np.load(f"{pred_path}/{pred_name}").astype("int16").transpose([2, 0, 1]))
        HQ_nrrd = itk.GetImageFromArray(np.load(f"{image_path}/{HQ_name}").astype("int16").transpose([2, 0, 1]))
        CE_nrrd = itk.GetImageFromArray(np.load(f"{CE_candidates[0]}").astype("int16").transpose([2, 0, 1]))

        itk.WriteImage(pred_nrrd, f"{save_path}/{pred_name[:-4]}.nrrd")
        itk.WriteImage(HQ_nrrd, f"{save_path}/{HQ_name[:-4]}.nrrd")
        itk.WriteImage(CE_nrrd, f"{save_path}/{CE_candidates[0][-15:][:-4]}.nrrd")


#-------------------------------------------------------------------------

if __name__ == "__main__":
    NRRDConv(
        "D:/ProjectImages/SyntheticContrastTest/Images",
        "C:/Users/roybo/Programming/PhD/007_CNN_Virtual_Contrast/test_pix2pix/H2_save280/predictions/",
        "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase3/output/H2_save280"
        )