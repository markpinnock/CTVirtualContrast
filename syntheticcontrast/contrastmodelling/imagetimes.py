import os
import SimpleITK as itk

from .util import process_time


FILE_PATH = "D:/ProjectImages/DICOM1/UCLH_12227856/CT-20151218"
subject = "T033A0"
images = ["T033A0AC011", "T033A0VC012", "T033A0HQ027", "T033A0HQ028", "T033A0HQ054", "T033A0HQ078"]
dicom_tags = {
    "study_time": "0008|0030",
    "series_time": "0008|0031",
    "acquisition_time": "0008|0032",
    "content_time": "0008|0033",
    "series_description": "0008|103e"
    }

slices = os.listdir(FILE_PATH)
img = itk.ReadImage(f"{FILE_PATH}/{slices[0]}")
study_time = process_time(img.GetMetaData(dicom_tags["study_time"]))

for i in range(len(slices)):
    img = itk.ReadImage(f"{FILE_PATH}/{slices[i]}")
    assert process_time(img.GetMetaData(dicom_tags["study_time"])) == study_time
    acq_time = process_time(img.GetMetaData(dicom_tags["acquisition_time"])) - study_time

    print(f"{acq_time:.2f}, {content_time:.2f}, {img.GetMetaData(dicom_tags['series_description'])}, {img.GetMetaData('0020|0011')}, {img.GetMetaData('0020|0012')}, {img.GetMetaData('0020|0013')}")

    # print(
    #     img.GetMetaData("0008|0033"), # Content time
    #     img.GetMetaData("0020|0011"), # Series number
    #     img.GetMetaData("0020|0012"), # Acquisition number
    #     img.GetMetaData("0020|0013"), # Instance number
    #     img.GetMetaData("0020|1041")) # Slice location

    # if i == 100:
    #     break
    # try:
    #     print(img.GetMetaData("0018|1042"))
    # except RuntimeError:
    #     pass
