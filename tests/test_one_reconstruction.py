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


def slice_at(path, thickness, acq=None, z=None, series=None):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = 2
    ds.SliceThickness = thickness
    if acq is not None:
        ds.AcquisitionNumber = acq
    if z is not None:
        ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    if series is not None:
        ds.SeriesNumber = series
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

    # a retake: the same range scanned twice, one acquisition after the other - the
    # later one is the scan, and the first is left out whole
    first = [slice_at(d / "retake" / f"a{i}.dcm", 0.6, acq=1, z=i) for i in range(10)]
    again = [slice_at(d / "retake" / f"b{i}.dcm", 0.6, acq=2, z=i) for i in range(10)]
    assert set(one_reconstruction(first + again)) == set(again)

    # a scan taken in two halves also has two acquisitions; both halves are kept
    top = [slice_at(d / "halves" / f"a{i}.dcm", 0.6, acq=1, z=i) for i in range(10)]
    low = [slice_at(d / "halves" / f"b{i}.dcm", 0.6, acq=2, z=10 + i) for i in range(10)]
    assert len(one_reconstruction(top + low)) == 20

    # a PACS export with every series in one folder: reading a choice back takes only
    # that series, not the scout and the dose report filed beside it
    import os
    os.environ["CT_DATA_ROOT"] = str(d)
    from ctseg import ct_paths
    ct_paths.DATA_ROOT = d
    flat = d / "proj" / "scans" / "CASE" / "DICOMOBJ"
    for i in range(2):
        slice_at(flat / f"0000{i}", 5.0, series=1)
    for i in range(10):
        slice_at(flat / f"001{i}", 0.6, series=2)
    for i in range(3):
        slice_at(flat / f"009{i}", 5.0, series=3)
    entry = {"snum": "2", "series_dir": "scans/CASE/DICOMOBJ"}
    files, _, snum = ct_paths.resolve_from_cache(entry, flat.parent, "proj")
    assert len(files) == 10 and snum == "2", len(files)

    # the chosen series at both ends of the folder, another one in the middle
    ends = d / "proj" / "scans" / "ENDS" / "DICOMOBJ"
    for i in range(5):
        slice_at(ends / f"000{i}", 0.6, series=2, z=i)
        slice_at(ends / f"900{i}", 0.6, series=2, z=5 + i)
        slice_at(ends / f"500{i}", 5.0, series=7, z=i)
    entry = {"snum": "2", "series_dir": "scans/ENDS/DICOMOBJ"}
    files, _, _ = ct_paths.resolve_from_cache(entry, ends.parent, "proj")
    assert len(files) == 10, len(files)
    # and the series preview sees the same ten
    from ctseg.ct_gui import _series_files
    assert len(_series_files(ends, "2")) == 10
    # a retake previews as the one scan that will be converted, not both stacked
    assert sorted(_series_files(d / "retake", "")) == sorted(again)

print("one reconstruction: ok")
