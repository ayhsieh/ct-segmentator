"""Does a task count as done only when it produced what was asked of it?

Fails if the old rule comes back: 'total.seg.nrrd exists' alone marked the task
finished, so a scan segmented before skull joined the subset was skipped for
ever - the mask never appeared, and every later run queued the same step to do
nothing again.

The same rule has to hold when nothing was named: a task.seg.nrrd beside an empty
folder is the same dead end, because the analyses read the separate masks and not
the summary. Three cases sat in that loop.

    python -m tests.test_task_done
"""
import tempfile
from pathlib import Path

from ctseg.segment_structures import task_done


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

    # the summary file on its own, with no masks beside it: AJ, AK and BC
    bare = case(tmp, "bare", ["total.seg.nrrd"])
    assert not task_done(bare, "total")
    assert not task_done(bare, "total", ["brain"])        # named, and not there

    # nothing named, and the masks are there: the summary is proof enough
    some = case(tmp, "some", ["total.seg.nrrd", "total/liver.nii.gz"])
    assert task_done(some, "total")

    half = case(tmp, "half", ["total.seg.nrrd", "total/brain.nii.gz"])
    assert task_done(half, "total", ["brain"])
    assert not task_done(half, "total", ["brain", "skull"])   # the 49 stuck cases

    whole = case(tmp, "whole",
                 ["total.seg.nrrd", "total/brain.nii.gz", "total/skull.nii.gz"])
    assert task_done(whole, "total", ["brain", "skull"])

    loose = case(tmp, "loose", ["total/brain.nii.gz"])     # no multilabel file
    assert not task_done(loose, "total", ["brain"])

print("task_done: ok")
