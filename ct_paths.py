"""Where a study group's folders live, and which series was chosen for a case.

Its own module because this is all a program needs in order to find results on disk or
answer from a recorded choice, and importing it from segment_structures drags in
totalsegmentator and torch - seconds on a warm machine, far worse on a cold one, and
several hundred megabytes of DLLs that can fail to load when Windows is short of
commit. Reading a folder of finished results should not load a deep learning framework.

segment_structures re-exports all of it, so `from segment_structures import seg_dir_for`
keeps working and there is only one definition of anything.
"""
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CT_DATA_ROOT", "ct_scans"))


def group_dir(group):
    """Where a study group's folder lives.

    Scans live under ct_scans/, which is gitignored as a directory - so no amount of
    `git add -A` can put patient data into the repository. Set CT_DATA_ROOT to keep
    them somewhere else entirely, an external drive say.

    A group still sitting in the repo root is honoured if it is there, so an older
    checkout keeps working; anything that does not exist yet resolves under
    ct_scans/, which is where new work should go."""
    p = Path(group)
    if p.is_absolute():
        return p
    under = DATA_ROOT / group
    return under if under.exists() or not p.exists() else p


def nifti_dir_for(group):
    """Directory under `<group>/converted_nifti_<group>/` for a study group."""
    return group_dir(group) / f"converted_nifti_{group}"


def seg_dir_for(group):
    """Directory under `<group>/total_segmentor_results_<group>/` for a study group."""
    return group_dir(group) / f"total_segmentor_results_{group}"


CACHE_FILE = Path(".series_selection_cache.json")


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


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
