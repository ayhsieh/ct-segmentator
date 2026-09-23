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

APP_ROOT = Path(__file__).resolve().parent.parent   # the top folder, not the package
DATA_ROOT = Path(os.environ.get("CT_DATA_ROOT", "ct_scans"))


def is_dicom_file(path):
    """Is this a DICOM file? The preamble check: 128 bytes, then "DICM".

    Here rather than in segment_structures because the interface needs it too, and
    importing that module for six lines would pull dicom2nifti and nibabel with it.
    """
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


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


def _inner(group, name, old_prefix):
    """`<group>/<name>`, moving a folder from the old naming into place the first time.

    Inner folders used to repeat the group's name - `converted_nifti_<group>` - so
    renaming a project meant renaming everything inside it as well. A checkout with
    projects from before still has those, and the first call after the update moves
    each into place, rather than finding an empty folder and reporting nothing done.
    """
    root = group_dir(group)
    d = root / name
    old = root / f"{old_prefix}_{root.name}"
    if not d.exists() and old.exists():
        try:
            old.rename(d)
        except OSError:           # the server and a job it started got there together
            pass
    return d


def nifti_dir_for(group):
    """The group's converted scans: `<group>/nifti/<case>/`."""
    return _inner(group, "nifti", "converted_nifti")


def seg_dir_for(group):
    """The group's segmentations and measurements: `<group>/results/<case>/`."""
    return _inner(group, "results", "total_segmentor_results")


# ------------------------------------------------------------- series choices
# Each group keeps its own, in `<group>/series.json`, with every path relative to the
# group's folder. Inside the group, so renaming or moving a project carries its choices
# with it; relative, so nothing recorded names the folder it happens to be in.
SERIES_FILE = "series.json"
# every group used to share one file at the top of the checkout
OLD_CACHE = APP_ROOT / ".series_selection_cache.json"


def series_file(group):
    return group_dir(group) / SERIES_FILE


def anchored(p, group):
    """A path as recorded: relative to the group's folder when it is inside it.

    Taken as written, not resolved. A case reached through a junction is recorded
    where the project sees it, not where the scan physically lives - so the key is the
    same however the case was reached, and survives the project being renamed.
    """
    p = Path(p).absolute()
    try:
        return p.relative_to(group_dir(group).absolute()).as_posix()
    except ValueError:
        return str(p)


def unanchored(p, group):
    """A recorded path as a real one: relative means relative to the group's folder."""
    p = Path(p)
    return p if p.is_absolute() else group_dir(group) / p


def load_cache(group):
    f = series_file(group)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return _from_old_cache(group)


def save_cache(group, cache):
    """Replaced rather than rewritten in place, so a reader never sees half a file."""
    f = series_file(group)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    os.replace(tmp, f)


def cache_get(cache, path, group):
    """The recorded choice for a case, or None."""
    return cache.get(anchored(path, group))


def _from_old_cache(group):
    """This group's choices, taken out of the one file every group used to share.

    Keys there were paths relative to the checkout, or absolute, and through a junction
    they were the scan's real location rather than the project's link to it - so each
    case is looked up under every name the old code could have written it as. Only the
    group's own cases come across; the old file is left as it was.
    """
    if not OLD_CACHE.exists():
        return {}
    old = json.loads(OLD_CACHE.read_text(encoding="utf-8"))
    root = group_dir(group)
    base = root / "scans" if (root / "scans").is_dir() else root
    out = {}
    for case in (d for d in base.iterdir() if d.is_dir()) if base.is_dir() else ():
        names = [str(case.absolute())]
        try:
            real = case.resolve()
            names += [str(real), str(real.relative_to(APP_ROOT))]
        except (OSError, ValueError):
            pass
        entry = next((old[n] for n in names if n in old), None)
        if isinstance(entry, dict) and entry.get("series_dir"):
            sd = Path(entry["series_dir"])
            sd = sd if sd.is_absolute() else APP_ROOT / sd
            out[anchored(case, group)] = {**entry, "series_dir": anchored(sd, group)}
    if out:
        save_cache(group, out)
    return out


def resolve_from_cache(entry, dicom_folder, group):
    """Try to resolve a work item from a cache entry (new dict format).
    Returns (files, desc, snum) or (None, None, None)."""
    if not isinstance(entry, dict) or not entry.get("series_dir"):
        return None, None, None
    series_dir = unanchored(entry["series_dir"], group)
    if not series_dir.is_dir():
        return None, None, None
    files = [str(f) for f in series_dir.iterdir() if f.is_file()]
    if not files:
        return None, None, None
    return files, entry.get("desc", ""), entry["snum"]
