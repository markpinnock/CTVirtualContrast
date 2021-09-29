import json
import numpy as np
import os
import SimpleITK as itk

from .util import process_time


def load_series(file_path):
    dicom_tags = {
        "study_time": "0008|0030",
        "series_time": "0008|0031",
        "acquisition_time": "0008|0032",
        "content_time": "0008|0033",
        "series_description": "0008|103e",
        "series_number": "0020|0011",
        "acquisition_number": "0020|0012",
        "instance_number": "0020|0013",
        "slice_location": "0020|1041",
        "contrast_time": "0018|1042"
        }

    slices = os.listdir(file_path)
    contrast_time = None
    max_series_number = 0

    img_details = {}

    for i in range(0, len(slices)):
        img = itk.ReadImage(f"{file_path}/{slices[i]}")
        series_number = int(img.GetMetaData(dicom_tags["series_number"]))

        if "2.0" in img.GetMetaData(dicom_tags["series_description"]) or "4.0" in img.GetMetaData(dicom_tags["series_description"]) or "0.5" in img.GetMetaData(dicom_tags["series_description"]) or "Body CT" in img.GetMetaData(dicom_tags["series_description"]) or "Thin Body" in img.GetMetaData(dicom_tags["series_description"]) or "Thick Body" in img.GetMetaData(dicom_tags["series_description"]):
            continue

        if series_number not in img_details.keys():
            if series_number > max_series_number:
                max_series_number = series_number
            
            try:
                img_details[series_number] = {
                    "description": img.GetMetaData(dicom_tags["series_description"]),
                    "time": process_time(img.GetMetaData(dicom_tags["acquisition_time"]))
                }
            except RuntimeError:
                print(f"Missing acquisition time for: {file_path} {series_number} {img.GetMetaData(dicom_tags['series_description'])}")
                continue

            try:
                contrast_time = process_time(img.GetMetaData(dicom_tags["contrast_time"]))
            except RuntimeError:
                pass
        
        else:
            assert img_details[series_number]["description"] == img.GetMetaData(dicom_tags["series_description"]), f"{file_path} {series_number} {img_details[series_number]['description']}, {img.GetMetaData(dicom_tags['series_description'])}"
            assert img_details[series_number]["time"] == process_time(img.GetMetaData(dicom_tags["acquisition_time"])), f"{file_path} {series_number} {img_details[series_number]['description']} {img_details[series_number]['time']}, {process_time(img.GetMetaData(dicom_tags['acquisition_time']))}"

            if contrast_time is not None:
                try:
                    assert contrast_time == process_time(img.GetMetaData(dicom_tags["contrast_time"])), f"{file_path} {series_number} {contrast_time}, {process_time(img.GetMetaData(dicom_tags['contrast_time']))}"
                except RuntimeError:
                    pass

    img_details["contrast_time"] = contrast_time

    return img_details


def load_subject(subject_dict):
    batch = subject_dict["batch"]
    subject_id = subject_dict["id"]
    study = subject_dict["study"]

    file_path = f"Z:/Raw_CT_Data/Renal_Project_Images/_Batch{batch}_Anon/{subject_id}/{study}"
    study_details = load_series(file_path)

    return study_details


def save_times(subjects):
    times = {}

    for key, val in subjects.items():
        details = load_subject(val)
        file_path = f"{file_path}/{key}"
        volumes = os.listdir(file_path)

        for vol in volumes:
           series_number = str(int(vol[-8:-5]))

           try:
               times[vol] = np.around(details[series_number]["time"] - details["contrast_time"], 1)
           except KeyError:
               continue
        
        if not os.path.exists(f"{SAVE_PATH}/{key}"):
            os.makedirs(f"{SAVE_PATH}/{key}")
        
        with open(f"{SAVE_PATH}/{key}/time.json", 'w'):
            json.dump(times, fp, indent=4)


if __name__ == "__main__":

    with open("Z:/Clean_CT_Data/id_pairs.json", 'r') as fp:
        id_pairs = json.load(fp)

    FILE_PATH = "Z:/Clean_CT_Data/Toshiba/Images"
    SAVE_PATH = "Z:/Clean_CT_Data/Toshiba/Times"
    
    save_times(id_pairs)
