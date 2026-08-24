#!/usr/bin/env python
"""
Create tilted variants of a sample case by rotating the NIfTI affines (world
orientation) about the volume centre - no voxel resampling, so the existing
TotalSegmentator masks stay valid and are copied over with the same rotation.
This isolates the geometry code: a correct pipeline must produce identical
volumes and boundary walls that stay perpendicular to the (now tilted) skull base.

    python make_tilted_sample.py --group sample_ct --case SAMPLE1 \
        --tilt SAMPLE1_TILT12:12:0 --tilt SAMPLE1_YAW20:0:0:20

Each --tilt is NEWNAME:pitch_deg[:roll_deg[:yaw_deg]] (pitch = head tilted back,
yaw = head turned to one side, which is the rotation that exposes any disagreement
between scanner-derived and anatomy-derived directions).
The cranial-landmark model output is NOT copied so predictions rerun on the
tilted world coordinates.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

from segment_structures import seg_dir_for, find_source_nifti, nifti_dir_for


def rotation(pitch_deg, roll_deg, yaw_deg=0.0):
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    y = np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tilt_affine(affine, shape, R):
    c = affine[:3, :3] @ (np.asarray(shape[:3], float) / 2.0) + affine[:3, 3]
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c
    return M @ affine


def retilt_file(src, dst, R):
    img = nib.load(str(src))
    new = nib.Nifti1Image(np.asarray(img.dataobj), tilt_affine(img.affine, img.shape, R),
                          img.header)
    new.header.set_sform(new.affine, code=1)
    new.header.set_qform(new.affine, code=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new, str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="sample_ct")
    ap.add_argument("--case", default="SAMPLE1")
    ap.add_argument("--tilt", action="append", required=True,
                    help="NEWNAME:pitch_deg[:roll_deg[:yaw_deg]]")
    args = ap.parse_args()

    ct = find_source_nifti(args.group, args.case)
    if ct is None:
        raise SystemExit(f"no converted NIfTI for {args.case}")
    src_seg = seg_dir_for(args.group) / args.case

    for spec in args.tilt:
        parts = spec.split(":")
        name, pitch = parts[0], float(parts[1])
        roll = float(parts[2]) if len(parts) > 2 else 0.0
        yaw = float(parts[3]) if len(parts) > 3 else 0.0
        R = rotation(pitch, roll, yaw)
        print(f"{name}: pitch {pitch:+.0f} deg, roll {roll:+.0f} deg, yaw {yaw:+.0f} deg")

        dst_ct = nifti_dir_for(args.group) / name / Path(ct).name
        retilt_file(Path(ct), dst_ct, R)
        print(f"  CT -> {dst_ct}")

        dst_seg = seg_dir_for(args.group) / name
        if dst_seg.exists():
            shutil.rmtree(dst_seg)
        n = 0
        for rel in ["total/brain.nii.gz", "total/skull.nii.gz"] + [
                f"brain_structures/{p.name}"
                for p in sorted((src_seg / "brain_structures").glob("*.nii.gz"))]:
            sp = src_seg / rel
            if sp.exists():
                retilt_file(sp, dst_seg / rel, R)
                n += 1
        print(f"  {n} mask(s) re-tilted -> {dst_seg}")


if __name__ == "__main__":
    main()
