"""Does a case's study date survive the way projects are actually laid out?

Fails if either bug that emptied the study_date column comes back:

  * the case folder was looked for at <group>/<case>, but a project made in the
    interface keeps its scans at <group>/scans/<case>, so every case came back
    "case folder not found" and the column was blank;
  * a recorded series choice is stored under a path anchored to this checkout, and
    ct_dates only matched absolute keys, so an answered case looked unanswered and
    the date was read from the whole folder instead of the chosen series.

    python -m tests.test_study_date
"""
import json
import os
import tempfile
from pathlib import Path


def dicom(path, date):
    """The smallest file pydicom will read a StudyDate out of."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyDate = date
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    os.environ["CT_DATA_ROOT"] = str(root)
    from ctseg import ct_paths
    ct_paths.DATA_ROOT = root
    ct_paths.APP_ROOT = root
    ct_paths.CACHE_FILE = root / ".series_selection_cache.json"
    from ctseg import ct_dates
    # a project made in the interface: scans live one level down, under scans/
    gui = root / "study" / "scans" / "CASE1"
    dicom(gui / "s2" / "1.dcm", "20260317")
    # and a group made on the command line: the case sits straight under the group
    cli = root / "old" / "CASE2"
    dicom(cli / "1.dcm", "20240101")

    assert ct_paths.case_dir_for("study", "CASE1") == gui
    assert ct_paths.case_dir_for("old", "CASE2") == cli

    d, note = ct_dates.study_date("study", "CASE1")
    assert d == "2026-03-17", (d, note)          # was "" - case folder not found
    assert ct_dates.study_date("old", "CASE2")[0] == "2024-01-01"

    # two studies exported into one case folder, and the chosen one is series 5
    both = root / "study" / "scans" / "CASE3"
    dicom(both / "s3" / "1.dcm", "20250101")
    dicom(both / "s5" / "1.dcm", "20250909")
    d, note = ct_dates.study_date("study", "CASE3")
    assert note.startswith("no series chosen"), note     # honest about guessing

    ct_paths.CACHE_FILE.write_text(json.dumps({
        ct_paths.anchored(both): {"snum": 5, "series_dir": ct_paths.anchored(both / "s5")},
    }))
    d, note = ct_dates.study_date("study", "CASE3")
    assert (d, note) == ("2025-09-09", ""), (d, note)    # was the folder's guess

print("study_date: ok")
