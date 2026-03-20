"""
Usage:
    python combine_segmentations.py /path/to/segmentation/folder
    python combine_segmentations.py /path/to/parent/folder  # processes all subfolders
"""

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path


def combine_segmentations(seg_folder):
    seg_folder = Path(seg_folder)
    nifti_files = sorted(seg_folder.glob("*.nii.gz"))

    if not nifti_files:
        print(f"  No NIfTI files found in '{seg_folder}'")
        return

    combined = None
    affine = None

    for nifti_file in nifti_files:
        img = nib.load(nifti_file)
        data = img.get_fdata()
        if combined is None:
            combined = np.zeros(data.shape, dtype=np.int16)
            affine = img.affine
        combined[data > 0] = 1

    combined_path = seg_folder / "combined.nii.gz"
    nib.save(nib.Nifti1Image(combined, affine), combined_path)
    print(f"Saved: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine segmentation NIfTI files into a single mask.")
    parser.add_argument("folders", nargs="+", help="Segmentation folder(s) or parent folder containing subfolders.")
    args = parser.parse_args()

    all_folders = []
    for folder in args.folders:
        p = Path(folder)
        if not p.is_dir():
            print(f"Not a directory: {folder}")
            continue
        niftis = list(p.glob("*.nii.gz"))
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if niftis:
            all_folders.append(p)
        elif subdirs:
            all_folders.extend(sorted(subdirs))
        else:
            all_folders.append(p)

    for folder in all_folders:
        combine_segmentations(folder)


if __name__ == "__main__":
    main()