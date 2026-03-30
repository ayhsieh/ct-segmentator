import pandas as pd
import os
import json

TOTAL_DIR = "total_segmentor_results"
NIFTI_DIR = "converted_nifti"

rows = {patient : {} for patient in os.listdir(TOTAL_DIR) if patient[0] != "." and not patient.endswith("zip") and not patient.endswith("txt")}



for patient in os.listdir(TOTAL_DIR):
    if patient == "." or patient.endswith("zip"): continue
    with open(os.path.join(TOTAL_DIR, patient, "statistics.json"), "r") as f:
        d = json.loads(f.read())
        for k, v in d.items():
            rows[patient][k+"_volume"] = v["volume_mm3"]

for patient in os.listdir(NIFTI_DIR):
    files =  os.listdir(os.path.join(NIFTI_DIR, patient))
    if not files: continue
    file = files[0].split("_")
    series = int(file[1])
    desc = "".join(file[2:])[:-7]
    rows[patient]["series_num"] = series
    rows[patient]["series name"] = desc

pd.DataFrame(rows).transpose().to_csv("segmentation_results.csv")
