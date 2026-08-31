"""Where a study group's folders live.

Its own module because these three functions are all that a program needs in order to
find results on disk, and importing them from segment_structures drags in
totalsegmentator and torch - seconds on a warm machine and far worse on a cold one.
Reading a folder of finished results should not load a deep learning framework.

segment_structures re-exports these, so `from segment_structures import seg_dir_for`
keeps working and there is only one definition of any of them.
"""
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
