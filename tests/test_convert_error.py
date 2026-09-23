"""Does a failed conversion say which kind of failure it was?

dicom2nifti raises MISSING_DICOM_FILES for a folder that is gone, a folder that is
empty, a folder whose files are not DICOM, and a folder it merely would not accept.
Those want four different things done about them, so the message has to tell them
apart - "MISSING_DICOM_FILES" on its own sends you looking in the wrong place.

    python -m tests.test_convert_error
"""
import tempfile
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from ctseg.segment_structures import why_convert_failed


def dicom(path):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SeriesNumber = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


boom = RuntimeError("MISSING_DICOM_FILES")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    gone = why_convert_failed(root / "nope", boom)
    assert "not there any more" in gone, gone

    (root / "empty").mkdir()
    assert "is empty" in why_convert_failed(root / "empty", boom)

    junk = root / "junk"
    junk.mkdir()
    (junk / "notes.txt").write_text("this is not a scan")
    (junk / "VERSION").write_text("1")
    msg = why_convert_failed(junk, boom)
    assert "none of the 2 checked are readable as DICOM" in msg, msg
    assert str(junk) in msg, msg          # says which folder, so it can be looked at

    real = root / "real"
    dicom(real / "1.dcm")
    dicom(real / "2.dcm")
    (real / "readme.txt").write_text("stray")
    msg = why_convert_failed(real, boom)
    assert "2 of the 3 checked read as DICOM" in msg, msg
    assert "MISSING_DICOM_FILES" in msg, msg   # the library's own words are kept

print("convert error: ok")
