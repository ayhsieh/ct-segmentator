"""
Usage:
    python segment_brain_structures.py /path/to/dicom/folder
"""

import os
import json
import argparse
import pydicom
import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings
from pathlib import Path
from collections import defaultdict

dicom2nifti_settings.disable_validate_orthogonal()
dicom2nifti_settings.disable_validate_slice_increment()
dicom2nifti_settings.disable_validate_multiframe_implicit()

NIFTI_OUTPUT_DIR = Path("converted_nifti")
TOTALSEG_OUTPUT_DIR = Path("total_segmentor_results")
CACHE_FILE = Path(".series_selection_cache.json")


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def sanitize_filename(name):
    """Replace characters that are invalid in filenames."""
    invalid = r'/\:*?"<>|'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name.strip()


def get_series(dicom_folder):
    """Returns dict mapping SeriesInstanceUID -> list of file paths."""
    series_map = defaultdict(list)
    for root, _, files in os.walk(dicom_folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                series_map[ds.SeriesInstanceUID].append(fpath)
            except Exception:
                pass
    return series_map


def get_series_number(files):
    """Get SeriesNumber from the first file in a series."""
    try:
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        return str(int(ds.SeriesNumber))
    except Exception:
        return None


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
    """Check if a folder contains DICOM files directly (not recursively)."""
    for f in folder.iterdir():
        if f.is_file() and not f.name.startswith("."):
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True)
                return True
            except Exception:
                pass
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
        # Subdirs are series folders of one study - treat parent as single folder
        return [p]
    else:
        # Subdirs are patient/study folders - process each separately
        print(f"\n'{p}' contains {len(subdirs)} subdirectories - processing each as a separate DICOM folder.")
        return sorted(subdirs)


def is_valid_series(files, desc):
    """Returns True if the series is not disqualified by exclusion rules."""
    words = desc.lower().split()
    if not words or len(files) < 20: return False
    return ("st" in words or words[-1] == "0.75") and "bone" not in words and "mpr" not in words


def prompt_series_selection(dicom_folder, all_series, candidates, folder_key, cache):
    """Prompt user to pick a series. Returns (files, desc) or None if skipped.
    all_series is a list of (files, desc, snum) tuples."""
    print(f"\n  Folder: {dicom_folder.name}")
    if candidates:
        print(f"  No single candidate auto-detected. Please pick one (or -1 to skip):")
    else:
        print(f"  No soft tissue series found. Please pick one manually (or -1 to skip):")

    for i, (files, desc, snum) in enumerate(all_series):
        marker = " *" if (files, desc) in candidates else ""
        print(f"    [{i+1}] [series {snum:>3}] [{len(files):>4} slices]  {desc or '(no description)'}{marker}")

    while True:
        try:
            choice = int(input(f"\n  Enter number (1-{len(all_series)}), or -1 to skip: ").strip())
            if choice == -1:
                print("  Skipping.")
                return None
            if 1 <= choice <= len(all_series):
                files, desc, snum = all_series[choice - 1]
                cache[folder_key] = snum
                save_cache(cache)
                print(f"  Selection series {snum} '{desc or '(no description)'}' cached for future use.")
                return files, desc
            else:
                print(f"  Please enter a number between 1 and {len(all_series)}, or -1 to skip.")
        except ValueError:
            print("  Invalid input, please enter a number.")


def plan_all_folders(all_folders, cache):
    """
    Pass 1: Scan all folders and resolve series selections.
    Auto-selects where confident, prompts for manual input at the end.
    Returns list of (dicom_folder, files, desc) tuples.
    """
    resolved = []
    needs_manual = []

    print("\n" + "="*60)
    print("PASS 1: Scanning all folders...")
    print("="*60)

    for dicom_folder in all_folders:
        dicom_folder = Path(dicom_folder)
        if not dicom_folder.is_dir():
            print(f"\nCould not find '{dicom_folder}' - skipping.")
            continue

        seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
        if seg_out.exists() and any(seg_out.iterdir()):
            print(f"\n[SKIP] '{dicom_folder.name}' - segmentation already exists.")
            continue

        folder_key = str(dicom_folder.resolve())

        # Scan series
        series_map = get_series(dicom_folder)
        series_info = []
        for uid, flist in series_map.items():
            ds = pydicom.dcmread(flist[0], stop_before_pixels=True)
            d = str(getattr(ds, "SeriesDescription", ""))
            snum = str(int(getattr(ds, "SeriesNumber", 0)))
            series_info.append((uid, flist, d, snum))

        files = None
        desc = None

        # Check cache first — matched by series number
        if folder_key in cache:
            cached_snum = cache[folder_key]
            print(f"\n[CACHE] '{dicom_folder.name}' - looking for series {cached_snum}...")
            for uid, flist, d, snum in series_info:
                if snum == cached_snum:
                    files, desc = flist, d
                    print(f"  -> Using cached: series {snum} '{d or '(no description)'}'")
                    break
            if files is None:
                print(f"  -> Cached series not found - falling back to auto selection.")

        # Simple exclusion-based selection
        if files is None:
            candidates = [(flist, d) for _, flist, d, _ in series_info if is_valid_series(flist, d)]
            series_info_simple = [(flist, d, snum) for _, flist, d, snum in series_info]

            if len(candidates) == 1:
                files, desc = candidates[0]
                snum = get_series_number(files)
                if snum:
                    cache[folder_key] = snum
                    save_cache(cache)
                print(f"\n[AUTO] '{dicom_folder.name}' - only candidate: '{desc}'")
            else:
                print(f"\n[MANUAL] '{dicom_folder.name}' - needs manual selection.")
                needs_manual.append((dicom_folder, folder_key, series_info_simple, candidates))
                continue

        resolved.append((dicom_folder, files, desc))

    # Prompt for all manual selections at once
    if needs_manual:
        print(f"\n" + "="*60)
        print(f"MANUAL SELECTION REQUIRED for {len(needs_manual)} folder(s):")
        print("="*60)

        for dicom_folder, folder_key, series_info_simple, candidates in needs_manual:
            result = prompt_series_selection(dicom_folder, series_info_simple, candidates, folder_key, cache)
            if result is not None:
                files, desc = result
                resolved.append((dicom_folder, files, desc))

    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dicom_folders", nargs="+")
    parser.add_argument("--skip-planning", action="store_true",
                        help="Skip series selection and only segment folders without existing output.")
    args = parser.parse_args()

    cache = load_cache()

    # Expand any parent folders into their subfolders
    all_folders = []
    for folder in args.dicom_folders:
        expanded = resolve_folders(folder)
        if not expanded:
            print(f"\nCould not find '{folder}' - skipping.")
        all_folders.extend(expanded)

    if args.skip_planning:
        # Skip planning — use cache where available, skip folders without existing segmentation
        work_items = []
        print("\n" + "="*60)
        print("SKIP PLANNING MODE: processing folders without existing segmentation...")
        print("="*60)
        for dicom_folder in all_folders:
            dicom_folder = Path(dicom_folder)
            seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
            if seg_out.exists() and any(seg_out.iterdir()):
                print(f"\n[SKIP] '{dicom_folder.name}' - segmentation already exists.")
                continue
            folder_key = str(dicom_folder.resolve())
            if folder_key not in cache:
                print(f"\n[SKIP] '{dicom_folder.name}' - no cached series selection, run without --skip-planning first.")
                continue
            # Resolve cached series
            series_map = get_series(dicom_folder)
            series_info = []
            for uid, flist in series_map.items():
                ds = pydicom.dcmread(flist[0], stop_before_pixels=True)
                d = str(getattr(ds, "SeriesDescription", ""))
                snum = str(int(getattr(ds, "SeriesNumber", 0)))
                series_info.append((flist, d, snum))
            cached_snum = cache[folder_key]
            matched = [(flist, d) for flist, d, snum in series_info if snum == cached_snum]
            if not matched:
                print(f"\n[SKIP] '{dicom_folder.name}' - cached series {cached_snum} not found.")
                continue
            files, desc = matched[0]
            print(f"\n[CACHED] '{dicom_folder.name}' - series {cached_snum} '{desc}'")
            work_items.append((dicom_folder, files, desc))
    else:
        # Pass 1: resolve all series selections
        work_items = plan_all_folders(all_folders, cache)

    if not work_items:
        print("\nNo folders to process. Exiting.")
        return

    # Pass 2: convert and segment
    print(f"\n" + "="*60)
    print(f"PASS 2: Processing {len(work_items)} folder(s)...")
    print("="*60)

    for dicom_folder, files, desc in work_items:
        print(f"\n{'='*60}")
        print(f"Processing: {dicom_folder}")
        print(f"{'='*60}")

        series_dir = str(Path(files[0]).parent)

        nifti_out = NIFTI_OUTPUT_DIR / dicom_folder.name
        nifti_out.mkdir(parents=True, exist_ok=True)
        snum = get_series_number(files) or "unknown"
        if desc:
            nifti_name = sanitize_filename(f"series_{snum}_{desc}")
        else:
            nifti_name = f"series_{snum}_unnamed"
        nifti_path = nifti_out / f"{nifti_name}.nii.gz"

        print(f"\nConverting: '{desc or '(no description)'}' -> {nifti_path}")
        dicom2nifti.dicom_series_to_nifti(series_dir, str(nifti_path))
        print("Done.")

        seg_out = TOTALSEG_OUTPUT_DIR / dicom_folder.name
        seg_out.mkdir(parents=True, exist_ok=True)
        cmd = f'TotalSegmentator -i "{nifti_path}" -o "{seg_out}" -ta brain_structures --statistics'
        print(f"\nRunning: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    main()