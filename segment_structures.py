import os
import re
import json
import argparse
import hashlib
import traceback

import nrrd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pydicom
import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings
import nibabel as nib
import numpy as np

from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti

dicom2nifti_settings.disable_validate_orthogonal()
dicom2nifti_settings.disable_validate_slice_increment()
dicom2nifti_settings.disable_validate_multiframe_implicit()

NIFTI_OUTPUT_DIR = Path("converted_nifti")
TOTALSEG_OUTPUT_DIR = Path("total_segmentor_results")
CACHE_FILE = Path(".series_selection_cache.json")

# All TotalSegmentator tasks (open license)
AVAILABLE_TASKS = [
    "total", "total_mr",
    "lung_vessels", "lung_vessels_LEGACY",
    "body",
    "cerebral_bleed",
    "hip_implant",
    "coronary_arteries", "coronary_arteries_LEGACY",
    "pleural_pericard_effusion",
    "liver_vessels",
    "oculomotor_muscles",
    "lung_nodules",
    "kidney_cysts",
    "brain_structures",
    "brain_structures_mr",
    "vertebrae_body",
    "face",
    "face_mr",
    "thigh_shoulder_muscles",
]

# Licensed tasks (free for non-commercial use)
LICENSED_TASKS = [
    "heartchambers_highres",
    "appendicular_bones", "appendicular_bones_mr",
    "tissue_types",
    "brain_aneurysm",
]

# Tags needed for series scanning
_SCAN_TAGS = ["SeriesInstanceUID", "SeriesNumber", "SeriesDescription"]

# Tags needed for scoring
_SCORE_TAGS = [
    "SeriesDescription", "SeriesNumber", "SliceThickness",
    "ConvolutionKernel", "ImageOrientationPatient", "WindowCenter",
]

# Known bone/sharp convolution kernels (case-insensitive substring matches)
BONE_KERNELS = ["bone", "b70", "b75", "b80", "h70", "h60", "d70", "edge", "sharp", "boneplus"]
# Known soft tissue / standard kernels
SOFT_KERNELS = ["soft", "standard", "b30", "b31", "b40", "h31", "h30", "j30", "j40",
                "j45", "i40", "c30", "hr40", "hp38", "stnd", "std"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Filename / DICOM helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name):
    """Replace characters that are invalid in filenames."""
    invalid = r'/\:*?"<>|'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name.strip()


def is_dicom_file(fpath):
    """Quick check if a file is DICOM by reading its preamble (128 bytes + 'DICM')."""
    try:
        with open(fpath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def get_series(dicom_folder):
    """Returns dict mapping SeriesInstanceUID -> list of (file_path, SeriesNumber, SeriesDescription).
    Reads only the minimal tags needed."""
    series_map = defaultdict(list)
    series_meta = {}  # uid -> (snum, desc)
    for root, _, files in os.walk(dicom_folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True,
                                     specific_tags=_SCAN_TAGS)
                uid = ds.SeriesInstanceUID
                series_map[uid].append(fpath)
                if uid not in series_meta:
                    snum = str(int(getattr(ds, "SeriesNumber", 0)))
                    desc = str(getattr(ds, "SeriesDescription", ""))
                    series_meta[uid] = (snum, desc)
            except Exception:
                pass
    return series_map, series_meta


def find_dicom_folder(name):
    """Search all DICOM* dirs in cwd for a folder matching name."""
    cwd = Path.cwd()
    for dicom_root in cwd.glob("DICOM*"):
        candidate = dicom_root / name
        if candidate.is_dir():
            print(f"Found: {candidate}")
            return candidate
    return None


def contains_dicom_files(folder):
    """Check if a folder contains DICOM files directly (not recursively).
    Uses fast preamble check before falling back to pydicom."""
    for f in folder.iterdir():
        if f.is_file() and not f.name.startswith("."):
            if is_dicom_file(str(f)):
                return True
    return False


def resolve_folders(folder):
    """
    Given a path, return a list of DICOM folders to process.
    - If it contains DICOM files directly -> single folder
    - If subdirs contain DICOMs directly (series subfolders) -> treat parent as one study
    - If subdirs contain further subdirs -> batch, process each subdir separately
    - Otherwise, try find_dicom_folder() as fallback.
    """
    p = Path(folder)
    if not p.is_dir():
        found = find_dicom_folder(folder)
        return [found] if found else []

    if contains_dicom_files(p):
        return [p]

    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if not subdirs:
        return [p]

    subdirs_with_dicoms = [d for d in subdirs if contains_dicom_files(d)]
    subdirs_with_subdirs = [d for d in subdirs if any(True for x in d.iterdir() if x.is_dir())]

    if subdirs_with_dicoms and not subdirs_with_subdirs:
        return [p]
    else:
        print(f"\n'{p}' contains {len(subdirs)} subdirectories - processing each as a separate DICOM folder.")
        return sorted(subdirs)


# ---------------------------------------------------------------------------
# Series selection helpers
# ---------------------------------------------------------------------------

def get_series_metadata(files):
    """Read DICOM metadata from a representative file in the series.
    Returns a dict with keys useful for scoring."""
    ds = pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=_SCORE_TAGS)

    desc = str(getattr(ds, "SeriesDescription", ""))
    snum = str(int(getattr(ds, "SeriesNumber", 0)))

    slice_thickness = None
    try:
        slice_thickness = float(ds.SliceThickness)
    except Exception:
        pass

    kernel = str(getattr(ds, "ConvolutionKernel", "")).lower()

    is_axial = False
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        row, col = iop[0:3], iop[3:6]
        normal_z = row[0]*col[1] - row[1]*col[0]
        is_axial = abs(normal_z) > 0.8
    except Exception:
        pass

    window_center = None
    try:
        wc = ds.WindowCenter
        window_center = float(wc[0]) if isinstance(wc, (list, pydicom.multival.MultiValue)) else float(wc)
    except Exception:
        pass

    return {
        "desc": desc,
        "snum": snum,
        "slice_thickness": slice_thickness,
        "kernel": kernel,
        "is_axial": is_axial,
        "window_center": window_center,
        "num_slices": len(files),
    }


def score_series(meta):
    """Score a series for likelihood of being the correct soft-tissue CT.
    Higher score = better candidate. Returns (score, reasons) tuple."""
    score = 0
    reasons = []
    desc_lower = meta["desc"].lower()

    # --- Disqualifiers ---
    if meta["num_slices"] < 20:
        score -= 100
        reasons.append("too few slices")

    if any(bk in meta["kernel"] for bk in BONE_KERNELS):
        score -= 50
        reasons.append(f"bone kernel '{meta['kernel']}'")

    for excl in ["bone", "mpr", "coronal", "sag", "sagittal"]:
        if excl in desc_lower:
            score -= 50
            reasons.append(f"'{excl}' in description")

    # --- Positive signals ---
    if any(sk in meta["kernel"] for sk in SOFT_KERNELS):
        score += 20
        reasons.append(f"soft kernel '{meta['kernel']}'")

    if meta["is_axial"]:
        score += 15
        reasons.append("axial orientation")

    if meta["slice_thickness"] is not None:
        if meta["slice_thickness"] <= 1.0:
            score += 25
            reasons.append(f"thin slices ({meta['slice_thickness']}mm)")
        elif meta["slice_thickness"] <= 2.0:
            score += 10
            reasons.append(f"moderate slices ({meta['slice_thickness']}mm)")
        elif meta["slice_thickness"] >= 4.0:
            score -= 10
            reasons.append(f"thick slices ({meta['slice_thickness']}mm)")

    if meta["window_center"] is not None:
        if 10 <= meta["window_center"] <= 100:
            score += 15
            reasons.append(f"soft tissue window (center={meta['window_center']:.0f})")
        elif meta["window_center"] > 300:
            score -= 20
            reasons.append(f"bone window (center={meta['window_center']:.0f})")

    if "st" in desc_lower.split() or "soft" in desc_lower:
        score += 10
        reasons.append("'ST/soft' in description")
    if "head" in desc_lower or "brain" in desc_lower:
        score += 10
        reasons.append("'head/brain' in description")
    if "3d" in desc_lower:
        score += 5
        reasons.append("'3D' in description")

    if meta["num_slices"] >= 200:
        score += 5
        reasons.append(f"high slice count ({meta['num_slices']})")

    return score, reasons


def prompt_series_selection(dicom_folder, scored_series, folder_key, cache):
    """Prompt user to pick a series. Returns (files, desc, snum) or None if skipped.
    scored_series is a list of (files, meta, score, reasons) tuples, sorted by score desc."""
    print(f"\n  Folder: {dicom_folder.name}")
    print(f"  Please pick a series (or -1 to skip):\n")

    for i, (files, meta, score, reasons) in enumerate(scored_series):
        thick_str = f"{meta['slice_thickness']}mm" if meta['slice_thickness'] else "?mm"
        kernel_str = meta['kernel'] or "?"
        orient_str = "axial" if meta['is_axial'] else "non-ax"
        reason_str = ", ".join(reasons) if reasons else "no signals"
        print(f"    [{i+1}] [series {meta['snum']:>3}] [{meta['num_slices']:>4} slices] "
              f"[{thick_str:>6}] [{kernel_str:>10}] [{orient_str}] "
              f"score={score:>4}  {meta['desc'] or '(no description)'}")
        print(f"         -> {reason_str}")

    while True:
        try:
            choice = int(input(f"\n  Enter number (1-{len(scored_series)}), or -1 to skip: ").strip())
            if choice == -1:
                print("  Skipping.")
                return None
            if 1 <= choice <= len(scored_series):
                files, meta, score, reasons = scored_series[choice - 1]
                cache[folder_key] = {
                    "snum": meta["snum"],
                    "desc": meta["desc"],
                    "series_dir": str(Path(files[0]).parent),
                }
                save_cache(cache)
                print(f"  Selection series {meta['snum']} '{meta['desc'] or '(no description)'}' cached.")
                return files, meta["desc"], meta["snum"]
            else:
                print(f"  Please enter a number between 1 and {len(scored_series)}, or -1 to skip.")
        except ValueError:
            print("  Invalid input, please enter a number.")


def resolve_from_cache(entry, dicom_folder):
    """Try to resolve a work item from a cache entry (new dict format).
    Returns (files, desc, snum) or (None, None, None)."""
    if not isinstance(entry, dict) or not entry.get("series_dir"):
        return None, None, None
    series_dir = Path(entry["series_dir"])
    if not series_dir.is_dir():
        return None, None, None
    files = [str(f) for f in series_dir.iterdir() if f.is_file()]
    if not files:
        return None, None, None
    return files, entry.get("desc", ""), entry["snum"]


def plan_all_folders(all_folders, cache, tasks, skip_manual=False, force_redo=False):
    """
    Pass 1: Scan all folders and resolve series selections.
    Uses DICOM metadata scoring to auto-select the best soft-tissue series.
    Falls back to manual selection if no clear winner.
    If skip_manual=True, folders needing manual selection are skipped.
    If force_redo=True, folders with existing output are re-processed.
    Returns list of (dicom_folder, files, desc, snum, tasks_to_run) tuples, where
    tasks_to_run is a subset of `tasks` that actually need to be run for that folder.
    """
    resolved = []
    needs_manual = []

    AUTO_SELECT_MIN_SCORE = 20
    AUTO_SELECT_MIN_GAP = 15

    print("\n" + "=" * 60)
    print(f"PASS 1: Scanning all folders (tasks: {', '.join(tasks)})...")
    print("  Auto-selection ENABLED")
    if skip_manual:
        print("  --skip-planning: folders needing manual selection will be skipped")
    if force_redo:
        print("  --force-redo: re-segmenting all folders regardless of existing output")
    print("=" * 60)

    for dicom_folder in all_folders:
        dicom_folder = Path(dicom_folder)
        if not dicom_folder.is_dir():
            print(f"\nCould not find '{dicom_folder}' - skipping.")
            continue

        # Figure out which tasks still need to run for this folder
        seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
        tasks_to_run = []
        for task in tasks:
            if force_redo:
                tasks_to_run.append(task)
                continue
            # We mark a task "done" if its multilabel file exists
            existing_ml = seg_out / f"{task}.nii.gz" if seg_out.exists() else None
            if existing_ml and existing_ml.exists():
                print(f"[SKIP] '{dicom_folder.name}' task '{task}' - already exists: {existing_ml.name}")
            else:
                tasks_to_run.append(task)

        if not tasks_to_run:
            continue

        folder_key = str(dicom_folder.resolve())

        # Fast path: cached entry with stored directory — zero DICOM reads
        if folder_key in cache:
            entry = cache[folder_key]
            files, desc, snum = resolve_from_cache(entry, dicom_folder)
            if files:
                print(f"\n[CACHE] '{dicom_folder.name}' - series {snum} '{desc or '(no description)'}'")
                resolved.append((dicom_folder, files, desc, snum, tasks_to_run))
                continue
            else:
                fmt = "old format" if not isinstance(entry, dict) else "directory missing"
                print(f"\n[CACHE] '{dicom_folder.name}' - {fmt}, doing full scan to update.")

        # Full scan: collect series and metadata
        series_map, series_meta = get_series(dicom_folder)
        scored_series = []
        for uid, flist in series_map.items():
            meta = get_series_metadata(flist)
            score, reasons = score_series(meta)
            scored_series.append((flist, meta, score, reasons))

        scored_series.sort(key=lambda x: x[2], reverse=True)

        if not scored_series:
            print(f"\n[SKIP] '{dicom_folder.name}' - no DICOM series found.")
            continue

        # Auto-selection
        best_files, best_meta, best_score, best_reasons = scored_series[0]
        second_score = scored_series[1][2] if len(scored_series) > 1 else -999
        gap = best_score - second_score

        if best_score >= AUTO_SELECT_MIN_SCORE and gap >= AUTO_SELECT_MIN_GAP:
            snum = best_meta["snum"]
            desc = best_meta["desc"]
            cache[folder_key] = {
                "snum": snum,
                "desc": desc,
                "series_dir": str(Path(best_files[0]).parent),
            }
            print(f"\n[AUTO] '{dicom_folder.name}' - series {snum} '{desc}' "
                  f"(score={best_score}, gap={gap})")
            print(f"  -> {', '.join(best_reasons)}")
            resolved.append((dicom_folder, best_files, desc, snum, tasks_to_run))
            continue

        # Manual selection needed
        print(f"\n[MANUAL] '{dicom_folder.name}' - needs manual selection.")
        needs_manual.append((dicom_folder, folder_key, scored_series, tasks_to_run))

    # Batch save cache after all auto-selections
    save_cache(cache)

    # Prompt for all manual selections at once
    if needs_manual:
        if skip_manual:
            print(f"\n" + "=" * 60)
            print(f"SKIPPED {len(needs_manual)} folder(s) requiring manual selection.")
            print("Run without --skip-planning to select series for these folders.")
            print("=" * 60)
        else:
            print(f"\n" + "=" * 60)
            print(f"MANUAL SELECTION REQUIRED for {len(needs_manual)} folder(s):")
            print("=" * 60)

            for dicom_folder, folder_key, scored_series, tasks_to_run in needs_manual:
                result = prompt_series_selection(dicom_folder, scored_series, folder_key, cache)
                if result is not None:
                    files, desc, snum = result
                    resolved.append((dicom_folder, files, desc, snum, tasks_to_run))

    return resolved


# ---------------------------------------------------------------------------
# Multilabel NIfTI helpers
# ---------------------------------------------------------------------------

def _deterministic_color(name):
    """Hash a structure name to a stable RGB triple in 0-255, avoiding very dark values."""
    h = hashlib.md5(name.encode("utf-8")).digest()
    r, g, b = h[0], h[1], h[2]
    # Lift dark values so segments are visible on dark backgrounds
    r = 60 + (r % 196)
    g = 60 + (g % 196)
    b = 60 + (b % 196)
    return r, g, b


def _strip_lr_suffix(name):
    """Strip _left/_right suffix so paired structures share a base color."""
    return re.sub(r'_(left|right)$', '', name)


def multilabel_to_segnrrd(seg_img, label_map, output_path):
    """Write a 3D Slicer .seg.nrrd file from a multilabel NIfTI image and label map.
    All segment names, colors, and metadata are baked into the nrrd header so no
    .ctbl or other companion files are needed.
    Left/right paired structures share a color."""
    output_path = Path(output_path)
    data = np.asarray(seg_img.dataobj, dtype=np.uint8)
    affine = seg_img.affine

    # Convert RAS -> LPS for Slicer
    lps_affine = affine.copy()
    lps_affine[0, :] *= -1
    lps_affine[1, :] *= -1

    origin = lps_affine[:3, 3].tolist()
    dir_i = lps_affine[:3, 0].tolist()
    dir_j = lps_affine[:3, 1].tolist()
    dir_k = lps_affine[:3, 2].tolist()

    # Build label -> name map
    name_by_label = {}
    for k, v in label_map.items():
        try:
            name_by_label[int(k)] = str(v)
        except (TypeError, ValueError):
            continue

    labels = sorted([int(l) for l in np.unique(data) if l > 0])

    header = {
        "type": "unsigned char",
        "space": "left-posterior-superior",
        "space directions": [dir_i, dir_j, dir_k],
        "space origin": origin,
        "kinds": ["domain", "domain", "domain"],
        "encoding": "gzip",
    }

    base_color_map = {}

    for seg_idx, label in enumerate(labels):
        name = name_by_label.get(label, f"label_{label}")
        base = _strip_lr_suffix(name)

        if base not in base_color_map:
            r, g, b = _deterministic_color(base)
            base_color_map[base] = f"{r/255:.3f} {g/255:.3f} {b/255:.3f}"
        color = base_color_map[base]

        nonzero = np.argwhere(data == label)
        if len(nonzero) > 0:
            mins = nonzero.min(axis=0)
            maxs = nonzero.max(axis=0)
            extent = f"{mins[0]} {maxs[0]} {mins[1]} {maxs[1]} {mins[2]} {maxs[2]}"
        else:
            extent = "0 0 0 0 0 0"

        header[f"Segment{seg_idx}_ID"] = name
        header[f"Segment{seg_idx}_Name"] = name
        header[f"Segment{seg_idx}_Layer"] = "0"
        header[f"Segment{seg_idx}_LabelValue"] = str(label)
        header[f"Segment{seg_idx}_Color"] = color
        header[f"Segment{seg_idx}_Extent"] = extent
        header[f"Segment{seg_idx}_NameAutoGenerated"] = "0"
        header[f"Segment{seg_idx}_ColorAutoGenerated"] = "0"

    nrrd.write(str(output_path), data, header, index_order='F')
    print(f"  Segmentation saved: {output_path} ({len(labels)} segments)")


def process_multilabel_output(multilabel_path, nifti_path=None, task_name=None):
    """Convert a multilabel NIfTI from TotalSegmentator into a single .seg.nrrd
    file (with all segment metadata baked in) and compute per-class statistics.

    The intermediate multilabel .nii.gz is deleted after successful conversion,
    leaving only the .seg.nrrd and .stats.json files."""
    multilabel_path = Path(multilabel_path)
    if not multilabel_path.exists():
        return

    try:
        seg_img, label_map = load_multilabel_nifti(str(multilabel_path))
    except Exception as e:
        print(f"  Could not read multilabel header from {multilabel_path.name}: {e}")
        return

    # Output .seg.nrrd named after the task (or the multilabel filename if not given)
    if task_name:
        segnrrd_path = multilabel_path.parent / f"{task_name}.seg.nrrd"
    else:
        stem = multilabel_path.name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        elif stem.endswith(".nii"):
            stem = stem[:-4]
        segnrrd_path = multilabel_path.parent / f"{stem}.seg.nrrd"

    try:
        multilabel_to_segnrrd(seg_img, label_map, segnrrd_path)
    except Exception as e:
        print(f"  Could not write .seg.nrrd: {e}")
        return

    # --- Compute statistics from the same in-memory data ---
    seg_data = np.asarray(seg_img.dataobj)
    voxel_dims = seg_img.header.get_zooms()[:3]
    voxel_vol_mm3 = float(np.prod(voxel_dims))

    source_data = None
    if nifti_path and Path(nifti_path).exists():
        try:
            source_data = np.asarray(nib.load(str(nifti_path)).dataobj)
            if source_data.shape != seg_data.shape:
                print(f"  Warning: source image shape {source_data.shape} "
                      f"!= segmentation shape {seg_data.shape}; skipping intensity stats.")
                source_data = None
        except Exception:
            pass

    stats = {}
    for label_value, structure_name in label_map.items():
        try:
            label_int = int(label_value)
        except (TypeError, ValueError):
            continue
        if label_int == 0:
            continue

        mask = seg_data == label_int
        voxel_count = int(np.count_nonzero(mask))
        volume_mm3 = round(voxel_count * voxel_vol_mm3, 2)

        entry = {
            "label": label_int,
            "volume_mm3": volume_mm3,
            "voxel_count": voxel_count,
        }
        if source_data is not None and voxel_count > 0:
            try:
                entry["mean_intensity"] = round(float(np.mean(source_data[mask])), 2)
            except Exception:
                pass

        stats[structure_name] = entry

    stats_name = task_name if task_name else multilabel_path.stem.replace(".nii", "")
    stats_path = multilabel_path.parent / f"{stats_name}.stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved: {stats_path}")

    # --- Delete the intermediate multilabel .nii.gz ---
    try:
        multilabel_path.unlink()
        print(f"  Removed intermediate file: {multilabel_path.name}")
    except Exception as e:
        print(f"  Could not remove intermediate file {multilabel_path.name}: {e}")


def find_source_nifti(dicom_folder_name):
    """Return the path to the converted NIfTI for a study, if one exists."""
    nifti_out = NIFTI_OUTPUT_DIR / dicom_folder_name
    if not nifti_out.is_dir():
        return None
    niftis = sorted(nifti_out.glob("*.nii.gz"))
    return niftis[0] if niftis else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment anatomical structures from DICOM using TotalSegmentator. "
                    "Produces one multilabel NIfTI per task, with class names stored in "
                    "the extended header and readable via load_multilabel_nifti().",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tasks:\n  " + "\n  ".join(AVAILABLE_TASKS) +
               f"\n\nLicensed tasks (free for non-commercial use):\n  " + "\n  ".join(LICENSED_TASKS)
    )
    parser.add_argument("dicom_folders", nargs="+",
                        help="DICOM folder(s) or parent folder containing subfolders.")
    parser.add_argument("--task", "-ta", required=True, nargs="+",
                        metavar="TASK",
                        help="TotalSegmentator task(s) to run. Accepts multiple, "
                             "e.g. --task total face head_muscles. "
                             "See the epilog for known tasks (not validated).")
    parser.add_argument("--skip-planning", action="store_true",
                        help="Skip manual series selection; only process cached and auto-selected folders.")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip segmentation; only recompute statistics from existing multilabel files.")
    parser.add_argument("--force-redo", action="store_true",
                        help="Re-segment all folders, even if output already exists.")
    parser.add_argument("--fast", action="store_true",
                        help="Use the lower-resolution (3mm) model for faster runtime.")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"],
                        help="Device for inference (default: gpu).")
    parser.add_argument("--license-number",
                        help="License number for commercial/restricted tasks.")
    args = parser.parse_args()

    # Deduplicate tasks while preserving order
    seen = set()
    tasks = [t for t in args.task if not (t in seen or seen.add(t))]

    cache = load_cache()

    # Expand any parent folders into their subfolders
    all_folders = []
    for folder in args.dicom_folders:
        expanded = resolve_folders(folder)
        if not expanded:
            print(f"\nCould not find '{folder}' - skipping.")
        all_folders.extend(expanded)

    # --stats-only: recompute stats from existing multilabel files
    if args.stats_only:
        print("\n" + "=" * 60)
        print("STATS-ONLY MODE")
        print("=" * 60)
        for dicom_folder in all_folders:
            name = Path(dicom_folder).name
            seg_out = TOTALSEG_OUTPUT_DIR / name
            if not seg_out.is_dir():
                print(f"\nNo segmentation output found for '{name}' - skipping.")
                continue

            source_nifti = find_source_nifti(name)
            multilabel_files = sorted(seg_out.glob("*.nii.gz"))
            if not multilabel_files:
                print(f"\nNo multilabel files in {seg_out} - skipping.")
                continue

            print(f"\nProcessing {name}:")
            for ml_file in multilabel_files:
                # Derive task name from filename (e.g. brain_structures.nii.gz -> brain_structures)
                task_name = ml_file.name
                if task_name.endswith(".nii.gz"):
                    task_name = task_name[:-7]
                elif task_name.endswith(".nii"):
                    task_name = task_name[:-4]
                print(f"  {ml_file.name}")
                process_multilabel_output(ml_file, source_nifti, task_name=task_name)
        return

    # Resolve work items (series selection)
    work_items = plan_all_folders(all_folders, cache, tasks,
                                   skip_manual=args.skip_planning,
                                   force_redo=args.force_redo)

    if not work_items:
        print("\nNo folders to process. Exiting.")
        return

    # Pass 2: convert + segment (one run per task)
    LOG_DIR = Path("error_logs")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_FILE = LOG_DIR / f"error_log_{timestamp}.txt"

    print(f"\n" + "=" * 60)
    print(f"PASS 2: Processing {len(work_items)} folder(s)...")
    print("=" * 60)

    for dicom_folder, files, desc, snum, tasks_to_run in work_items:
        try:
            print(f"\n{'=' * 60}")
            print(f"Processing: {dicom_folder}")
            print(f"Tasks: {', '.join(tasks_to_run)}")
            print(f"{'=' * 60}")

            series_dir = str(Path(files[0]).parent)

            # --- Convert DICOM to NIfTI (once per study) ---
            nifti_out = NIFTI_OUTPUT_DIR / dicom_folder.name
            nifti_out.mkdir(parents=True, exist_ok=True)
            if desc:
                nifti_stem = sanitize_filename(f"series_{snum}_{desc}")
            else:
                nifti_stem = f"series_{snum}_unnamed"
            nifti_path = nifti_out / f"{nifti_stem}.nii.gz"

            if not nifti_path.exists():
                print(f"\nConverting: '{desc or '(no description)'}' -> {nifti_path}")
                dicom2nifti.dicom_series_to_nifti(series_dir, str(nifti_path))
                print("Conversion done.")
            else:
                print(f"\nUsing existing NIfTI: {nifti_path}")

            # --- Run TotalSegmentator for each requested task ---
            seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
            seg_out.mkdir(parents=True, exist_ok=True)

            for task in tasks_to_run:
                multilabel_path = seg_out / f"{task}.nii.gz"
                print(f"\n--- Running TotalSegmentator task '{task}' ---")
                print(f"    input:  {nifti_path}")
                print(f"    output: {multilabel_path}")

                kwargs = dict(
                    input=str(nifti_path),
                    output=str(multilabel_path),
                    task=task,
                    ml=True,             # <-- single multilabel file per task
                    fast=args.fast,
                    device=args.device,
                    statistics=False,    # we compute our own below
                )
                if args.license_number:
                    kwargs["license_number"] = args.license_number

                totalsegmentator(**kwargs)

                # --- Compute our own per-class statistics from the multilabel file ---
                print(f"    Computing statistics...")
                process_multilabel_output(multilabel_path, nifti_path, task_name=task)

        except Exception as e:
            error_msg = (
                f"[ERROR] {dicom_folder}\n"
                f"  Series: {snum} - {desc or '(no description)'}\n"
                f"  Tasks:  {', '.join(tasks_to_run)}\n"
                f"  Error:  {e}\n"
                f"  Traceback:\n{traceback.format_exc()}\n"
            )
            print(f"\n[ERROR] Failed on '{dicom_folder.name}': {e}")
            print(f"  Logged to {LOG_FILE}. Continuing...\n")
            with open(LOG_FILE, "a") as log:
                log.write(error_msg)

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()