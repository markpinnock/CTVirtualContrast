import os
import scipy.stats as stat

from .contrast_test import calc_contrast


#-------------------------------------------------------------------------

def get_model_results(real_path, pred_path, models):
    contrasts = {}

    for model in models:
        contrasts[model] = calc_contrast(real_path, f"{pred_path}/{model}")

    return contrasts


#-------------------------------------------------------------------------

def get_mannwhitneyu():


#-------------------------------------------------------------------------

if __name__ == "__main__":
    real_path = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output/Real"
    pred_path = f"C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/007_CNN_Virtual_Contrast/Phase2/output"
    models = [m for m in os.listdir(pred_path) if os.path.isdir(f"{pred_path}/{m}") and m != "Real"]

    contrast = get_model_results(real_path, pred_path, models)
    print(stat.mannwhitneyu(contrast["2_save230"]["AC"]["Ao"], contrast["2_save230"]["AP"]["Ao"]))
