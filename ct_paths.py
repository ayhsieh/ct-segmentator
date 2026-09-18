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

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("CT_DATA_ROOT", "ct_scans"))


def anchored(p):
    """A path as it should be recorded: relative to this checkout when it is inside it.

    Everything a project owns now lives under it, so recording absolute paths ties the
    cache to one machine and one folder name. Written relative, the whole checkout can
    be moved, copied to another computer or handed to someone else and every recorded
    series choice still points at the right folder.
    """
    p = Path(p)
    try:
        return str(p.resolve().relative_to(APP_ROOT))
    except (ValueError, OSError):
        return str(p)


def unanchored(p):
    """A recorded path as a real one.

    Relative means relative to this file's folder, never to the working directory. Some
    old entries were written relative to wherever the program happened to start, which
    resolved from the repository root and nowhere else."""
    p = Path(p)
    return p if p.is_absolute() else APP_ROOT / p


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


def case_dir_for(group, case):
    """Where a case's DICOM folder is.

    A project made in the interface keeps its scans in `<group>/scans/<case>`; a group
    made on the command line puts the case straight under the group. Both are honoured,
    so a caller only has to know the group and the case name."""
    root = group_dir(group)
    under = root / "scans" / case
    return under if under.is_dir() else root / case


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


def cache_keys(path):
    """The names one case can be recorded under, best first.

    Three of them. The anchored one is what gets written now. The absolute one is what
    older entries hold, and what a case outside this checkout still needs. The literal
    one covers a link whose target is elsewhere - a project can hold junctions, so a
    case has two honest paths and which one a caller holds depends on how it got there.
    """
    p = Path(path)
    out = [anchored(p)]
    try:
        out.append(str(p.resolve()))
    except OSError:
        pass
    out.append(str(p))
    seen, keys = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def cache_get(cache, path):
    """The recorded choice for a case, under either of its names, or None."""
    for k in cache_keys(path):
        if k in cache:
            return cache[k]
    return None


def resolve_from_cache(entry, dicom_folder):
    """Try to resolve a work item from a cache entry (new dict format).
    Returns (files, desc, snum) or (None, None, None)."""
    if not isinstance(entry, dict) or not entry.get("series_dir"):
        return None, None, None
    series_dir = unanchored(entry["series_dir"])
    if not series_dir.is_dir():
        return None, None, None
    files = [str(f) for f in series_dir.iterdir() if f.is_file()]
    if not files:
        return None, None, None
    return files, entry.get("desc", ""), entry["snum"]
