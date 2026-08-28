#!/usr/bin/env python
"""
One wide CSV row per case: every structure volume from every task that has been run.

Reads each case's <task>.stats.json files under the group's results folder and pivots
them into columns named <task>_<structure>_volume, plus the series number/name and
slice count read from the converted NIfTI. Nothing is recomputed - this only reads
what segment_structures.py already wrote, so it is safe to run any time.

    python produce_table.py --group fossa
    python produce_table.py --group fossa --out fossa_table.csv
"""
import argparse
import json
import os

import nibabel as nib
import pandas as pd

from segment_structures import seg_dir_for, nifti_dir_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="fossa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    total_dir = seg_dir_for(args.group)
    nifti_dir = nifti_dir_for(args.group)
    if not total_dir.is_dir():
        raise SystemExit(f"no such results folder: {total_dir}. Run "
                         "segment_structures.py on this group first.")

    cases = [p.name for p in total_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")]
    rows = {case: {} for case in cases}

    for case in cases:
        case_dir = total_dir / case
        for f in case_dir.glob("*.stats.json"):
            task = f.name[: -len(".stats.json")]
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for structure, info in d.items():
                if isinstance(info, dict) and "volume_mm3" in info:
                    rows[case][f"{task}_{structure}_volume"] = info["volume_mm3"]

    for case in cases:
        nifti_case_dir = nifti_dir / case
        if not nifti_case_dir.is_dir():
            continue
        files = sorted(nifti_case_dir.glob("*.nii.gz"))
        if not files:
            continue
        fname = files[0].name
        parts = fname.split("_")
        if len(parts) >= 3:
            try:
                rows[case]["series_num"] = int(parts[1])
            except ValueError:
                pass
            rows[case]["series_name"] = "".join(parts[2:])[: -len(".nii.gz")]
        try:
            rows[case]["num_slices"] = nib.load(str(files[0])).shape[2]
        except Exception:
            rows[case]["num_slices"] = None

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "case"
    out = args.out or str(total_dir / f"{args.group}_table.csv")
    df.to_csv(out)
    print(f"{len(cases)} case(s) -> {out}")


if __name__ == "__main__":
    main()
