"""Does a project survive being renamed, and does an old one come across intact?

A project's inner folders used to repeat its name, and every series choice lived in
one shared file keyed by paths that included that name - so renaming the folder lost
the choices, and a checkout from before the change would have looked empty. Checked
here: the old folders move into place, the old choices carry over, and a renamed
project still finds both.

    python -m tests.test_project_layout
"""
import json
import os
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    os.environ["CT_DATA_ROOT"] = str(root)
    from ctseg import ct_paths
    ct_paths.DATA_ROOT = root
    ct_paths.APP_ROOT = root
    ct_paths.OLD_CACHE = root / ".series_selection_cache.json"

    # a project laid out the old way
    g = root / "study"
    (g / "scans" / "C1" / "s2").mkdir(parents=True)
    (g / "scans" / "C1" / "s2" / "1.dcm").write_bytes(b"")
    (g / "converted_nifti_study" / "C1").mkdir(parents=True)
    (g / "total_segmentor_results_study" / "C1").mkdir(parents=True)
    # and its choice in the old shared file, the way the old code wrote it
    ct_paths.OLD_CACHE.write_text(json.dumps({
        str(Path("study") / "scans" / "C1"): {
            "snum": "2", "desc": "head", "series_dir": str(Path("study") / "scans" / "C1" / "s2")},
        str(Path("elsewhere") / "C9"): {"snum": "1", "series_dir": "x"},   # not this project
    }))

    # the inner folders move into place the first time they are asked for
    assert ct_paths.nifti_dir_for("study") == g / "nifti" and (g / "nifti" / "C1").is_dir()
    assert ct_paths.seg_dir_for("study") == g / "results" and (g / "results" / "C1").is_dir()
    assert not (g / "converted_nifti_study").exists()

    # the project's own choice comes across, relative to the project; nobody else's does
    cache = ct_paths.load_cache("study")
    assert cache == {"scans/C1": {"snum": "2", "desc": "head", "series_dir": "scans/C1/s2"}}, cache
    assert (g / "series.json").exists()

    # renamed: one folder, and everything inside it still answers
    g.rename(root / "renamed")
    case = ct_paths.case_dir_for("renamed", "C1")
    entry = ct_paths.cache_get(ct_paths.load_cache("renamed"), case, "renamed")
    files, desc, snum = ct_paths.resolve_from_cache(entry, case, "renamed")
    assert snum == "2" and files and Path(files[0]).name == "1.dcm", (entry, files)
    assert ct_paths.seg_dir_for("renamed") == root / "renamed" / "results"

print("project layout: ok")
