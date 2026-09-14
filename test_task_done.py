"""Does a task count as done only when it produced what was asked of it?

Fails if the old rule comes back: 'total.seg.nrrd exists' alone marked the task
finished, so a scan segmented before skull joined the subset was skipped for
ever - the mask never appeared, and every later run queued the same step to do
nothing again.

    python test_task_done.py
"""
import tempfile
from pathlib import Path

from segment_structures import task_done


def case(tmp, name, files):
    d = Path(tmp) / name
    for f in files:
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_bytes(b"")
    return d

with tempfile.TemporaryDirectory() as tmp:
    nothing = Path(tmp) / "nothing"
    assert not task_done(nothing, "total")
    assert not task_done(nothing, "total", ["brain"])

    bare = case(tmp, "bare", ["total.seg.nrrd"])
    assert task_done(bare, "total")                       # nothing named, file is proof
    assert not task_done(bare, "total", ["brain"])        # named, and not there

    half = case(tmp, "half", ["total.seg.nrrd", "total/brain.nii.gz"])
    assert task_done(half, "total", ["brain"])
    assert not task_done(half, "total", ["brain", "skull"])   # the 49 stuck cases

    whole = case(tmp, "whole",
                 ["total.seg.nrrd", "total/brain.nii.gz", "total/skull.nii.gz"])
    assert task_done(whole, "total", ["brain", "skull"])

    loose = case(tmp, "loose", ["total/brain.nii.gz"])     # no multilabel file
    assert not task_done(loose, "total", ["brain"])

print("task_done: ok")
