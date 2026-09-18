#!/usr/bin/env python
"""
One wide CSV row per case: every number any part of the pipeline has worked out.

Reads each case's stats files under the group's results folder and pivots them into
columns. Three kinds of file, three shapes:

  <task>.stats.json     a volume per structure -> <task>_<structure>_volume
  brain_icv.stats.json  brain and intracranial volume -> brain_icv_<name>
  *fossae_simple.*.json the three fossa compartments -> fossae_<compartment>_<name>,
                        and the outside of the skull in mm -> outer_<name>

plus the series number/name and slice count read from the converted NIfTI. Nothing is
recomputed - this only reads what has already been written, so it is safe to run any
time, and it is the only thing that writes a table.

    python produce_table.py --group fossa
    python produce_table.py --group fossa --out fossa_structure_volumes_ml.csv

--select takes a JSON file naming what to keep: {"task": ["structure", ...]}, where an
empty list means the whole task, and "_scan" covers the series and slice-count columns.

Without it the table holds everything except the whole body. The "total" task is run
here only for its brain mask, but early cases were segmented before that was true and
their stats files hold all 117 structures, spleen to sacrum. Carrying those would give
a table where the body columns are filled in for the cases that happen to be old, so
total contributes its brain and nothing else unless --select asks for more.
"""
import argparse
import json
import os
import pathlib

from ct_dates import study_date
from ct_paths import group_dir, seg_dir_for, nifti_dir_for


def project_cases(group):
    """Every case the project lists, in the order it lists them.

    A case that was set aside, or that never converted, has no results folder - so
    reading the results folder alone quietly drops it, and a case missing from a cohort
    table is exactly the one worth seeing. Empty for a group made on the command line,
    which has no project.json; the results folder covers that.
    """
    try:
        text = (group_dir(group) / "project.json").read_text(encoding="utf-8")
        return [c["case"] for c in json.loads(text).get("cases", []) if c.get("case")]
    except Exception:
        return []


def case_notes(group):
    """Whatever was written about each case on the project screen, by case name.

    Read straight off project.json. Going through the server module would drag the
    whole interface, and torch behind it, into a CSV build."""
    try:
        text = (group_dir(group) / "project.json").read_text(encoding="utf-8")
        return json.loads(text).get("notes", {})
    except Exception:
        return {}


# Everything millilitres, so any two columns can be read against each other. The unit
# is said once, in the file's name, rather than in every heading.
BRAIN_ICV_KEYS = ["brain_ml", "icv_ml", "removed_from_brain_ml",
                  "brain_clipped_to_icv_ml"]


def stats_files(case_dir):
    """The stats files for a case, with superseded ones left out.

    Results written before output filenames carried the case name are still on disk
    beside the current ones - fossae_simple.stats.json next to CASE_fossae_simple.
    They describe the same thing and the plain one is older, so reading both means the
    order of a glob decides which numbers a table gets. The one that names the case
    wins; where there is no such file, the plain one is all there is and is used.
    """
    files = sorted(case_dir.glob("*.stats.json"))
    named = {f.name for f in files}
    stem = "_".join(str(case_dir.name).split()) + "_"
    return [f for f in files
            if not (stem + f.name) in named]


def read_stats(task, d):
    """(task, column) and value for every number in one stats file.

    The task a column is filed under is not always the file's name: a fossa file is
    named after its case, and a task file's own name is the task. Yielding the pair
    keeps that decision here, where the shapes are, rather than in each caller.
    """
    if not isinstance(d, dict):
        return
    # a linear-only file carries outer_mm and no compartments at all
    if (task.endswith("fossae_simple") or "compartments" in d
            or "outer_mm" in d):
        if isinstance(d.get("icv_ml"), (int, float)):
            yield ("fossae", "icv_ml"), d["icv_ml"]
        # the outside of the head, filed on its own so a table of sizes does not have
        # to carry the compartment volumes to get them
        for k, v in (d.get("outer_mm") or {}).items():
            if isinstance(v, (int, float)):     # "points" is coordinates, not a number
                yield ("outer", k), v
        for comp, v in (d.get("compartments") or {}).items():
            if not isinstance(v, dict):
                continue
            if isinstance(v.get("ml"), (int, float)):
                yield ("fossae", f"{comp}_ml"), v["ml"]
            if isinstance(v.get("percent_of_icv"), (int, float)):
                yield ("fossae", f"{comp}_pct_icv"), v["percent_of_icv"]
        return
    if task == "brain_icv":
        for k in BRAIN_ICV_KEYS:
            if isinstance(d.get(k), (int, float)):
                yield ("brain_icv", k), d[k]
        return
    for structure, info in d.items():
        if isinstance(info, dict) and "volume_mm3" in info:
            yield (task, f"{structure}_volume"), round(info["volume_mm3"] / 1000.0, 3)


def main():
    # here rather than at the top: ct_gui imports read_stats to build the column
    # picker, and the server has no use for pandas
    import nibabel as nib
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="fossa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--select", default=None, metavar="FILE",
                    help="JSON file of {task: [structure, ...]} to keep; an empty list "
                         "means the whole task, and _scan the series columns")
    args = ap.parse_args()

    keep = None
    if args.select:
        keep = {k: set(v) for k, v in json.loads(
            pathlib.Path(args.select).read_text()).items()}

    def wanted(task, structure=None):
        if keep is None:
            # total is run for its brain mask alone; cases segmented before that
            # was true carry all 117 whole-body structures, and a column filled in
            # only for those is worse than no column. --select can still ask.
            return task != "total" or structure in (None, "brain_volume")
        if task not in keep:
            return False
        return structure is None or not keep[task] or structure in keep[task]

    total_dir = seg_dir_for(args.group)
    nifti_dir = nifti_dir_for(args.group)

    # Every case in the project, plus any results folder the project does not list.
    # A row with nothing but a name and a note is the honest record of a case that was
    # set aside or never finished; leaving it out makes the cohort look complete.
    done = sorted(p.name for p in total_dir.iterdir()
                  if p.is_dir() and not p.name.startswith(".")) if total_dir.is_dir() else []
    listed = project_cases(args.group)
    cases = listed + [c for c in done if c not in set(listed)]
    if not cases:
        raise SystemExit(f"no cases for {args.group}: neither {total_dir} nor a "
                         "project.json lists any. Run segment_structures.py first.")
    rows = {case: {} for case in cases}

    for case in cases:
        case_dir = total_dir / case
        for f in stats_files(case_dir):
            task = f.name[: -len(".stats.json")]
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for col, val in read_stats(task, d):
                t, name = col
                if wanted(t, name):
                    rows[case][f"{t}_{name}"] = val

    # Always, whatever columns were asked for: a note is the reason a row looks the
    # way it does, and it is no use in a file that dropped it.
    notes = case_notes(args.group)
    for case in cases:
        rows[case]["note"] = notes.get(case, "")

    for case in cases if wanted("_scan") else []:
        # the date of the series this case was segmented from, not of whatever DICOM
        # the folder happens to hold first
        d, note = study_date(args.group, case)
        rows[case]["study_date"] = d
        # Only when there IS a date and it might be the wrong one. With no date the
        # blank already says so, and saying it again put a sentence on every row of a
        # cohort that had simply not been run yet.
        if d and note:
            rows[case]["study_date_note"] = note
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

    # reindex: from_dict does not keep the insertion order once the rows carry
    # different keys, and the table should read in the order the project lists its
    # cases - the same order the Cases tab shows.
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(cases)
    df.index.name = "case"
    if "note" in df.columns:          # next to the name it belongs to, not at the end
        df = df[["note"] + [c for c in df.columns if c != "note"]]
    out = args.out or str(total_dir / f"{args.group}_structure_volumes_ml.csv")
    df.to_csv(out)
    print(f"{len(cases)} case(s) -> {out}")


if __name__ == "__main__":
    main()
