#!/usr/bin/env python
"""Turn Slicer markups into the landmark CSV the pipeline reads.

Landmarks are placed by hand in Slicer, which saves them as a .mrk.json. The pipeline
reads <group>/points/<case>.csv. Doing that conversion by hand is where it goes wrong:
the file has to be named after the case - exactly one match, or it is ignored without
saying so - and the coordinates have to stay in LPS.

    python markups_to_points.py --group STUDY SAMPLE1.mrk.json
    python markups_to_points.py --group STUDY --case AB points/*.mrk.json
    python markups_to_points.py --group STUDY --check

`--check` names every case whose landmarks the pipeline would not find, and why.
"""
import argparse
import json
import sys
from pathlib import Path

from ct_paths import group_dir
from segment_fossae import MANUAL_LABELS, _norm_name

HEADER = ("# Markups outputs\n"
          "# CoordinateSystem = LPS\n")


def read_markups(path):
    """(label, x, y, z) for every control point, in LPS."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for m in d.get("markups", []):
        lps = (m.get("coordinateSystem", "LPS").upper() == "LPS")
        for c in m.get("controlPoints", []):
            x, y, z = (float(v) for v in c["position"])
            if not lps:                      # RAS in the file, LPS on the way out
                x, y = -x, -y
            out.append((str(c.get("label", "")).strip(), x, y, z))
    return out


def write_points(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(HEADER)
        for label, x, y, z in rows:
            f.write(f"{label},{x:.4f},{y:.4f},{z:.4f}\n")


def case_of(path):
    """The case a markups file is for, from its name, the way the loader reads it."""
    return Path(path).name.split(".")[0].replace("_Points", "").replace("Points", "")


def check(group):
    """Which cases have landmarks the pipeline will actually use."""
    pdir = group_dir(group) / "points"
    if not pdir.is_dir():
        print(f"no points folder: {pdir}")
        return 1
    seen = {}
    for f in sorted(pdir.glob("*.csv")):
        seen.setdefault(_norm_name(f.stem.replace("Points", "")), []).append(f.name)
    bad = 0
    for key, files in sorted(seen.items()):
        if len(files) > 1:
            # the loader refuses rather than guess which timepoint a case means
            print(f"AMBIGUOUS {key}: {len(files)} files - {', '.join(files)}")
            bad += 1
    for f in sorted(pdir.glob("*.csv")):
        labels = {r[0] for r in read_csv_labels(f)}
        unknown = labels - set(MANUAL_LABELS)
        used = labels & set(MANUAL_LABELS)
        note = f"{len(used)} used"
        if unknown:
            note += f", ignored: {', '.join(sorted(unknown))}"
        print(f"  {f.name}: {note}")
        if not used:
            bad += 1
    return 1 if bad else 0


def read_csv_labels(path):
    import csv
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#") or len(row) < 4:
                continue
            if row[0].strip().lower() == "label":
                continue
            out.append((row[0].strip(),))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="Slicer .mrk.json files")
    ap.add_argument("--group", required=True)
    ap.add_argument("--case", help="the case name to file these under "
                                   "(default: taken from each file's name)")
    ap.add_argument("--check", action="store_true",
                    help="report what the pipeline would find, and change nothing")
    args = ap.parse_args()

    if args.check:
        return check(args.group)
    if not args.files:
        ap.error("give at least one .mrk.json, or --check")
    if args.case and len(args.files) > 1:
        ap.error("--case names one case, so it takes one file")

    pdir = group_dir(args.group) / "points"
    pdir.mkdir(parents=True, exist_ok=True)
    for src in args.files:
        rows = read_markups(src)
        case = args.case or case_of(src)
        kept = [r for r in rows if r[0] in MANUAL_LABELS]
        dropped = sorted({r[0] for r in rows} - set(MANUAL_LABELS))
        out = pdir / f"{case}.csv"
        write_points(rows, out)          # everything is written; the loader filters
        print(f"{Path(src).name} -> {out}  ({len(kept)} of {len(rows)} points used"
              + (f", ignored: {', '.join(dropped)}" if dropped else "") + ")")
        if not kept:
            print(f"  !! none of these labels is one the pipeline reads. It wants: "
                  f"{', '.join(MANUAL_LABELS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
