"""Does a multi-valued convolution kernel come out as text a person can read?

Fails if the old bug comes back: ConvolutionKernel is VR SH with VM 1-n, so pydicom
hands back a MultiValue, and str() on that prints a Python list. An ordinary Siemens
head scan then showed up in the series picker as

    Kernel   ['j30s', '2']

and, worse, none of the SOFT_KERNELS substrings matched it, so the series scored as if
its kernel were unknown.

    python -m tests.test_series_meta
"""
import tempfile
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from ctseg.segment_structures import SOFT_KERNELS, get_series_metadata


def scan(path, kernel):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SeriesNumber = 4
    ds.SeriesDescription = "HEAD ROUTINE"
    ds.SliceThickness = 1.0
    ds.ConvolutionKernel = kernel
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)

    multi = get_series_metadata([scan(d / "multi.dcm", ["J30s", "2"])])
    assert multi["kernel"] == "j30s 2", multi["kernel"]     # was "['j30s', '2']"
    assert any(k in multi["kernel"] for k in SOFT_KERNELS), multi["kernel"]

    one = get_series_metadata([scan(d / "one.dcm", "BONEPLUS")])
    assert one["kernel"] == "boneplus", one["kernel"]       # a plain string is untouched

    none = get_series_metadata([scan(d / "none.dcm", "")])
    assert none["kernel"] == "", none["kernel"]             # absent stays empty, not "none"

print("series meta: ok")
