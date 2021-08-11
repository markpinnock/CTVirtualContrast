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


def load_subjects(id_pairs):
    subjects = {}

    for subject in id_pairs.keys():
        print(subject)

        if subject == "T025A1":
            file_path = f"Z:/Raw_CT_Data/Renal_Project_Images/_Batch2_Anon/{id_pairs[subject][0]}"
        else:
            file_path = f"Z:/Raw_CT_Data/Renal_Project_Images/_Batch1_Anon/{id_pairs[subject][0]}"

        studies = os.listdir(file_path)
        subjects[subject] = load_series(f"{file_path}/{studies[id_pairs[subject][1]]}")

    return subjects


def save_times(subjects):
    times = {}

    for key, val in subjects.items():
        file_path = f"Z:/Clean_CT_Data/Toshiba/Images/{key}"
        volumes = os.listdir(file_path)

        for vol in volumes:
            series_number = str(int(vol[-8:-5]))

            try:
                times[vol] = np.around(val[series_number]["time"] - val["contrast_time"], 1)
            except KeyError:
                continue

    return times


if __name__ == "__main__":
    # FILE_PATH = "D:/ProjectImages/DICOM1/UCLH_12227856/CT-20151218"
    FILE_PATH = "C:/ProjectDICOM/DICOM1/CT-20161210/"
    subject = "T033A0"
    images = ["T033A0AC011", "T033A0VC012", "T033A0HQ027", "T033A0HQ028", "T033A0HQ054", "T033A0HQ078"]

    id_pairs = {
        "T002A1": ["UCLH_01177351", 1],
        "T004A0": ["UCLH_01493856", 0],
        "T005A0": ["UCLH_01568304", 0],
        "T006A0": ["UCLH_01800867", 0],
        "T006A1": ["UCLH_01800867", 1],
        "T007A0": ["UCLH_01856974", 0],
        "T009A0": ["UCLH_02173725", 0],
        "T011A0": ["UCLH_03666794", 0],
        "T013A0": ["UCLH_04136428", 0],
        "T014A0": ["UCLH_04228827", 0],
        "T016A0": ["UCLH_05450311", 0],
        "T017A0": ["UCLH_05667514", 0],
        "T018A0": ["UCLH_05815905", 0],
        "T019A0": ["UCLH_07039934", 0],
        "T020A0": ["UCLH_08288605", 0],
        "T021A0": ["UCLH_08440465", 0],
        "T022A0": ["UCLH_08470270", 0],
        "T023A0": ["UCLH_08933783", 0],
        "T024A0": ["UCLH_09099884", 0],
        "T025A1": ["UCLH_10453550", 0],
        "T026A0": ["UCLH_11107604", 0],
        "T027A0": ["UCLH_11192578", 0],
        "T028A0": ["UCLH_11349911", 0],
        "T029A0": ["UCLH_11461915", 0],
        "T030A0": ["UCLH_11647475", 0],
        "T031A0": ["UCLH_11700946", 0],
        "T032A0": ["UCLH_11815331", 0],
        "T033A0": ["UCLH_12227856", 0],
        "T035A0": ["UCLH_12293559", 0],
        "T036A0": ["UCLH_12564192", 0],
        "T037A0": ["UCLH_12577934", 0],
        "T038A0": ["UCLH_12647573", 0],
        "T040A0": ["UCLH_13821133", 0],
        "T041A0": ["UCLH_14057035", 0]
    }

    # subjects = load_subjects(id_pairs)
    # print(subjects)
    
    with open("syntheticcontrast/contrastmodelling/recovery.json", 'r') as fp:
        subjects = json.load(fp)

    times = save_times(subjects)
    print(times)
    
    with open("syntheticcontrast/contrastmodelling/times.json", 'w') as fp:
        json.dump(times, fp, indent=4)
