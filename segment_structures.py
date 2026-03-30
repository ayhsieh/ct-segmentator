"""
Segment anatomical structures from DICOM using TotalSegmentator and combine results.

Usage:
    python segment_and_combine.py --task brain_structures /path/to/dicom/folder
    python segment_and_combine.py --task total /path/to/parent/folder
    python segment_and_combine.py --task lung_vessels folder1 folder2
    python segment_and_combine.py --task brain_structures --skip-planning folder
"""

import os
import json
import argparse
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pydicom
import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings
import nibabel as nib
import numpy as np

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

ALL_TASKS = AVAILABLE_TASKS + LICENSED_TASKS

# Tasks where automatic head CT series selection applies
BRAIN_TASKS = {"brain_structures", "brain_structures_mr", "cerebral_bleed",
               "oculomotor_muscles", "face", "face_mr", "brain_aneurysm"}

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


def plan_all_folders(all_folders, cache, task, skip_manual=False, force_redo=False):
    """
    Pass 1: Scan all folders and resolve series selections.
    For brain-related tasks, uses DICOM metadata scoring to auto-select.
    For other tasks, always prompts for manual selection.
    If skip_manual=True, folders needing manual selection are skipped.
    If force_redo=True, folders with existing output are re-processed.
    Returns list of (dicom_folder, files, desc, snum) tuples.
    """
    resolved = []
    needs_manual = []

    AUTO_SELECT_MIN_SCORE = 20
    AUTO_SELECT_MIN_GAP = 15

    print("\n" + "=" * 60)
    print(f"PASS 1: Scanning all folders (task: {task})...")
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

        seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
        if not force_redo:
            if seg_out.exists() and any(seg_out.iterdir()) and (seg_out / "statistics.json").exists():
                print(f"\n[SKIP] '{dicom_folder.name}' - segmentation already exists.")
                continue
            elif seg_out.exists() and any(seg_out.iterdir()):
                print(f"\n[REDO] '{dicom_folder.name}' - output exists but statistics.json missing, will re-segment.")

        folder_key = str(dicom_folder.resolve())

        # Fast path: cached entry with stored directory — zero DICOM reads
        if folder_key in cache:
            entry = cache[folder_key]
            files, desc, snum = resolve_from_cache(entry, dicom_folder)
            if files:
                print(f"\n[CACHE] '{dicom_folder.name}' - series {snum} '{desc or '(no description)'}'")
                resolved.append((dicom_folder, files, desc, snum))
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
        if True:
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
                resolved.append((dicom_folder, best_files, desc, snum))
                continue

        # Manual selection needed
        print(f"\n[MANUAL] '{dicom_folder.name}' - needs manual selection.")
        needs_manual.append((dicom_folder, folder_key, scored_series))

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

            for dicom_folder, folder_key, scored_series in needs_manual:
                result = prompt_series_selection(dicom_folder, scored_series, folder_key, cache)
                if result is not None:
                    files, desc, snum = result
                    resolved.append((dicom_folder, files, desc, snum))

    return resolved


# ---------------------------------------------------------------------------
# Combine segmentation masks and compute statistics
# ---------------------------------------------------------------------------

def compute_statistics(seg_folder, nifti_path=None):
    """Compute volume (mm³) and mean intensity for each segmentation mask.
    Saves results to statistics_custom.json in the segmentation folder.
    If nifti_path is provided, also computes mean intensity within each mask."""
    seg_folder = Path(seg_folder)
    nifti_files = sorted(seg_folder.glob("*.nii.gz"))

    if not nifti_files:
        return

    # Load the source image for intensity stats if available
    source_data = None
    if nifti_path and Path(nifti_path).exists():
        try:
            source_data = np.asarray(nib.load(str(nifti_path)).dataobj)
        except Exception:
            pass

    stats = {}
    for nifti_file in nifti_files:
        if nifti_file.name in ("combined.nii.gz",):
            continue

        img = nib.load(nifti_file)
        data = np.asarray(img.dataobj)
        mask = data > 0

        # Voxel volume in mm³ from the header
        voxel_dims = img.header.get_zooms()[:3]
        voxel_vol_mm3 = float(np.prod(voxel_dims))

        voxel_count = int(np.count_nonzero(mask))
        volume_mm3 = round(voxel_count * voxel_vol_mm3, 2)

        structure_name = nifti_file.name.replace(".nii.gz", "")
        entry = {
            "volume_mm3": volume_mm3,
            "voxel_count": voxel_count,
        }

        if source_data is not None and voxel_count > 0:
            try:
                entry["mean_intensity"] = round(float(np.mean(source_data[mask])), 2)
            except Exception:
                pass

        stats[structure_name] = entry

    stats_path = seg_folder / "statistics_custom.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Custom statistics saved: {stats_path}")


def combine_segmentations(seg_folder):
    """Combine all NIfTI segmentation masks in a folder into a single binary mask."""
    seg_folder = Path(seg_folder)
    nifti_files = sorted(seg_folder.glob("*.nii.gz"))

    if not nifti_files:
        print(f"  No NIfTI files found in '{seg_folder}' - nothing to combine.")
        return

    combined = None
    affine = None

    for nifti_file in nifti_files:
        if nifti_file.name == "combined.nii.gz":
            continue
        img = nib.load(nifti_file)
        data = np.asarray(img.dataobj)
        if combined is None:
            combined = np.zeros(data.shape, dtype=np.uint8)
            affine = img.affine
        combined[data > 0] = 1

    if combined is not None:
        combined_path = seg_folder / "combined.nii.gz"
        nib.save(nib.Nifti1Image(combined, affine), combined_path)
        print(f"  Combined mask saved: {combined_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment anatomical structures from DICOM using TotalSegmentator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available tasks:\n  " + "\n  ".join(AVAILABLE_TASKS) +
               f"\n\nLicensed tasks (free for non-commercial use):\n  " + "\n  ".join(LICENSED_TASKS)
    )
    parser.add_argument("dicom_folders", nargs="+",
                        help="DICOM folder(s) or parent folder containing subfolders.")
    parser.add_argument("--task", "-ta", required=True, choices=ALL_TASKS,
                        help="TotalSegmentator task to run (e.g., brain_structures, total, lung_vessels).")
    parser.add_argument("--skip-planning", action="store_true",
                        help="Skip manual series selection; only process cached and auto-selected folders.")
    parser.add_argument("--combine-only", action="store_true",
                        help="Skip segmentation; only combine existing results in the output directory.")
    parser.add_argument("--force-redo", action="store_true",
                        help="Re-segment all folders, even if output already exists.")
    args = parser.parse_args()

    cache = load_cache()

    # Expand any parent folders into their subfolders
    all_folders = []
    for folder in args.dicom_folders:
        expanded = resolve_folders(folder)
        if not expanded:
            print(f"\nCould not find '{folder}' - skipping.")
        all_folders.extend(expanded)

    # --combine-only: just run the combination and statistics steps on existing outputs
    if args.combine_only:
        print("\n" + "=" * 60)
        print("COMBINE-ONLY MODE")
        print("=" * 60)
        for dicom_folder in all_folders:
            name = Path(dicom_folder).name
            seg_out = TOTALSEG_OUTPUT_DIR / name
            if seg_out.is_dir():
                print(f"\nCombining masks in: {seg_out}")
                combine_segmentations(seg_out)
                # Try to find the source NIfTI for intensity stats
                nifti_out = NIFTI_OUTPUT_DIR / name
                nifti_path = None
                if nifti_out.is_dir():
                    niftis = list(nifti_out.glob("*.nii.gz"))
                    if niftis:
                        nifti_path = str(niftis[0])
                print(f"Computing statistics from masks...")
                compute_statistics(seg_out, nifti_path)
            else:
                print(f"\nNo segmentation output found for '{name}' - skipping.")
        return

    # Resolve work items (series selection)
    work_items = plan_all_folders(all_folders, cache, args.task,
                                   skip_manual=args.skip_planning,
                                   force_redo=args.force_redo)

    if not work_items:
        print("\nNo folders to process. Exiting.")
        return

    # Pass 2: convert, segment, and combine
    LOG_DIR = Path("error_logs")
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LOG_FILE = LOG_DIR / f"error_log_{timestamp}.txt"

    print(f"\n" + "=" * 60)
    print(f"PASS 2: Processing {len(work_items)} folder(s)...")
    print("=" * 60)

    for dicom_folder, files, desc, snum in work_items:
        try:
            print(f"\n{'=' * 60}")
            print(f"Processing: {dicom_folder}")
            print(f"{'=' * 60}")

            series_dir = str(Path(files[0]).parent)

            # --- Convert DICOM to NIfTI ---
            nifti_out = NIFTI_OUTPUT_DIR / dicom_folder.name
            nifti_out.mkdir(parents=True, exist_ok=True)
            if desc:
                nifti_name = sanitize_filename(f"series_{snum}_{desc}")
            else:
                nifti_name = f"series_{snum}_unnamed"
            nifti_path = nifti_out / f"{nifti_name}.nii.gz"

            print(f"\nConverting: '{desc or '(no description)'}' -> {nifti_path}")
            dicom2nifti.dicom_series_to_nifti(series_dir, str(nifti_path))
            print("Conversion done.")

            # --- Run TotalSegmentator ---
            seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
            seg_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                "TotalSegmentator",
                "-i", str(nifti_path),
                "-o", str(seg_out),
                "-ta", args.task,
                "--statistics",
            ]
            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(f"TotalSegmentator exited with code {result.returncode}")

            # --- Combine segmentation masks ---
            print(f"\nCombining segmentation masks in: {seg_out}")
            combine_segmentations(seg_out)

            # --- Compute our own statistics ---
            print(f"Computing statistics from masks...")
            compute_statistics(seg_out, nifti_path)

        except Exception as e:
            error_msg = (
                f"[ERROR] {dicom_folder}\n"
                f"  Series: {snum} - {desc or '(no description)'}\n"
                f"  Error: {e}\n"
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