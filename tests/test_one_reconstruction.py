"""Does a series holding two reconstructions convert as one volume?

Some scanners write two reconstructions of the same acquisition under one series
number - a 2 mm set beside a 6 mm set over the same range. They share a
SeriesInstanceUID, so the scan groups them together and both land in the chosen
folder. A converter given both sees the slice spacing change halfway down, treats
the series as 4D, and refuses it with the famously unhelpful MISSING_DICOM_FILES.

    python -m tests.test_one_reconstruction
"""
import tempfile
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from ctseg.segment_structures import one_reconstruction


def slice_at(path, thickness):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SliceThickness = thickness
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)

    # one reconstruction: nothing is dropped, and the list comes back as it went in
    plain = [slice_at(d / "one" / f"{i}.dcm", 1.0) for i in range(5)]
    assert one_reconstruction(plain) == plain

    # two: the thin one is kept whole and the coarse one is dropped whole
    thin = [slice_at(d / "two" / f"a{i}.dcm", 2.0) for i in range(114)]
    coarse = [slice_at(d / "two" / f"b{i}.dcm", 6.0) for i in range(38)]
    kept = one_reconstruction(thin + coarse)
    assert len(kept) == 114, len(kept)
    assert set(kept) == set(thin), "kept the wrong reconstruction"

    # a file with no thickness at all does not win over a real one
    odd = [slice_at(d / "odd" / f"c{i}.dcm", 3.0) for i in range(4)]
    noth = d / "odd" / "junk.dcm"
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    FileDataset(str(noth), {}, file_meta=meta, preamble=b"\0" * 128).save_as(
        str(noth), enforce_file_format=True)
    kept = one_reconstruction(odd + [str(noth)])
    assert set(kept) == set(odd), kept

print("one reconstruction: ok")
