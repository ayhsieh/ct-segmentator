import pandas as pd
import os
import json
import nibabel as nib

TOTAL_DIR = "total_segmentor_results"
NIFTI_DIR = "converted_nifti"

rows = {patient: {} for patient in os.listdir(TOTAL_DIR)
        if patient[0] != "." and not patient.endswith("zip") and not patient.endswith("txt")}

for patient in rows:
    patient_dir = os.path.join(TOTAL_DIR, patient)
    if not os.path.isdir(patient_dir):
        continue

    # Read every {task}.stats.json file in this patient's folder
    for fname in os.listdir(patient_dir):
        if not fname.endswith(".stats.json"):
            continue
        task = fname[:-len(".stats.json")]
        with open(os.path.join(patient_dir, fname), "r") as f:
            d = json.load(f)
        for structure, info in d.items():
            rows[patient][f"{task}_{structure}_volume"] = info["volume_mm3"]

for patient in rows:
    nifti_patient_dir = os.path.join(NIFTI_DIR, patient)
    if not os.path.isdir(nifti_patient_dir):
        continue
    files = os.listdir(nifti_patient_dir)
    if not files:
        continue
    fname = files[0]
    parts = fname.split("_")
    series = int(parts[1])
    desc = "".join(parts[2:])[:-7]
    rows[patient]["series_num"] = series
    rows[patient]["series_name"] = desc

    try:
        nifti_path = os.path.join(nifti_patient_dir, fname)
        img = nib.load(nifti_path)
        rows[patient]["num_slices"] = img.shape[2]
    except Exception:
        rows[patient]["num_slices"] = None

pd.DataFrame(rows).transpose().to_csv("segmentation_results.csv")