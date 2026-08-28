#!/usr/bin/env python
"""Build synthetic segmented projects for testing the viewers.

    python samples/make_fixtures.py

Shapes with geometry you can assert on, rather than a scan you have to squint at -
and nothing here is anyone's data. Two projects:

    SHAPES_TEST / PHANTOM   a sphere, a cube and a slab running off the edge of the
                            volume, plus a nested ICV/brain pair. Anisotropic voxels
                            and a non-RAS affine on purpose, so orientation and aspect
                            bugs show up as wrong numbers rather than a wrong-looking
                            picture. The slab is there to catch surfaces that are not
                            closed where the field of view clipped them.
    MANY_TEST / SIXTY       sixty small blobs, for the 3D display cap and for timing
                            what a task with a lot of structures actually costs.

Delete the projects from the tool's own list when you are done with them.
"""
import importlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
ss = importlib.import_module("segment_structures")
bi = importlib.import_module("brain_icv")


def project(group, case):
    p = APP / "projects" / group
    nii = p / f"converted_nifti_{group}" / case
    seg = p / f"total_segmentor_results_{group}" / case
    for d in (nii, seg):
        d.mkdir(parents=True, exist_ok=True)
    return p, nii, seg


def register(p, group, case, blurb):
    (p / "project.json").write_text(json.dumps(
        {"name": group, "source": str(p), "created": "2026-01-01 00:00",
         "description": blurb,
         "cases": [{"case": case, "path": str(p / case),
                    "series_root": str(p / case)}]}, indent=2))


def shapes():
    group, case = "SHAPES_TEST", "PHANTOM"
    p, nii, seg = project(group, case)
    nx, ny, nz = 120, 120, 100
    zx, zy, zz = 1.0, 1.0, 1.5                      # deliberately anisotropic
    aff = np.diag([-zx, zy, zz, 1.0])               # and deliberately not RAS
    aff[:3, 3] = [60.0, -60.0, -75.0]
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")

    ct = np.full((nx, ny, nz), -1000, np.int16)
    lab = np.zeros((nx, ny, nz), np.uint8)
    sphere = ((gx - 30.) ** 2 + (gy - 60.) ** 2 + (gz - 50.) ** 2) < 20.0 ** 2
    cube = ((gx >= 70) & (gx < 90) & (gy >= 30) & (gy < 50)
            & (gz >= 20) & (gz < 40))
    slab = (gx < 12) & (gy >= 80) & (gy < 100) & (gz >= 60) & (gz < 80)
    lab[sphere], lab[cube], lab[slab] = 1, 2, 3
    ct[sphere], ct[cube], ct[slab] = 60, 900, 300
    ss.multilabel_to_segnrrd(
        nib.Nifti1Image(lab, aff),
        {1: "sphere_r20", 2: "cube_20mm", 3: "edge_slab"},
        seg / "brain_structures.seg.nrrd")

    # The 4-D two-layer file, which is the shape brain_icv writes and the one most
    # likely to be read back wrongly: every segment has LabelValue 1 and only the
    # layer tells them apart.
    outer = ((gx - 60.) ** 2 + (gy - 60.) ** 2 + (gz - 50.) ** 2) < 34.0 ** 2
    inner = ((gx - 60.) ** 2 + (gy - 60.) ** 2 + (gz - 50.) ** 2) < 22.0 ** 2
    bi.write_two_layer_segnrrd([("intracranial_volume", outer), ("brain", inner)],
                               aff, seg / "brain_icv.seg.nrrd")
    ct[outer & (ct == -1000)] = 40
    nib.save(nib.Nifti1Image(ct, aff), str(nii / f"{case}_ct.nii.gz"))
    register(p, group, case, "synthetic shapes for the viewers - no patient data")
    print(f"  {group}: sphere 40x40x60 mm, cube 20x20x30 mm, slab on the -x edge, "
          "nested ICV/brain")


def many():
    group, case = "MANY_TEST", "SIXTY"
    p, nii, seg = project(group, case)
    n = 100
    aff = np.diag([-0.8, 0.8, 0.8, 1.0])
    aff[:3, 3] = [40, -40, -40]
    gx, gy, gz = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    ct = np.full((n, n, n), -1000, np.int16)
    lab = np.zeros((n, n, n), np.uint8)
    rng = np.random.default_rng(7)                  # fixed, so runs are comparable
    names = {}
    for k in range(1, 61):
        c, r = rng.integers(12, 88, 3), int(rng.integers(5, 9))
        blob = ((gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - c[2]) ** 2) < r * r
        lab[blob & (lab == 0)] = k
        ct[blob] = 300
        names[k] = f"blob_{k:02d}"
    ss.multilabel_to_segnrrd(nib.Nifti1Image(lab, aff), names,
                             seg / "total.seg.nrrd")
    nib.save(nib.Nifti1Image(ct, aff), str(nii / f"{case}_ct.nii.gz"))
    register(p, group, case, "60 synthetic blobs - display cap and timing")
    print(f"  {group}: 60 blobs")


def dicom():
    """A DICOM series that has never been converted, for the convert-on-demand path.

    Written out rather than borrowed from a real study for the obvious reason, and it
    exercises the bit that matters: series scoring has to pick this one, and the
    converter has to produce the spacing and orientation the tags describe.
    """
    import datetime
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    group, case = "DICOM_TEST", "RAW"
    p = APP / "projects" / group
    raw = p / case
    raw.mkdir(parents=True, exist_ok=True)
    for f in raw.glob("*.dcm"):
        f.unlink()
    nx = ny = 96
    nslices = 24
    px = py = 0.8
    thick = 2.0
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    for k in range(nslices):
        a = np.full((ny, nx), -1000, np.int16)
        r = 34 - abs(k - nslices // 2) * 0.8
        if r > 4:
            head = ((gx - nx / 2) ** 2 + (gy - ny / 2) ** 2) < r * r
            a[head.T] = 40
            a[(head & (((gx - nx / 2) ** 2 + (gy - ny / 2) ** 2) > (r - 3) ** 2)).T] = 900
        meta = Dataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(raw / f"s{k:03d}.dcm"), {}, file_meta=meta,
                         preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.FrameOfReferenceUID = frame
        ds.PatientName, ds.PatientID = "PHANTOM^SYNTHETIC", "SYNTH001"
        ds.Modality, ds.SeriesNumber, ds.InstanceNumber = "CT", 2, k + 1
        ds.SeriesDescription = "Head Synthetic 2.0 Hr40"
        ds.StudyDate = ds.SeriesDate = datetime.date(2026, 1, 1).strftime("%Y%m%d")
        ds.Rows, ds.Columns = ny, nx
        ds.PixelSpacing = [py, px]
        ds.SliceThickness = ds.SpacingBetweenSlices = thick
        ds.ImagePositionPatient = [-nx * px / 2, -ny * py / 2, k * thick]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.ConvolutionKernel = "Hr40"
        ds.PixelData = a.tobytes()
        ds.save_as(str(raw / f"s{k:03d}.dcm"), write_like_original=False)
    register(p, group, case, "synthetic DICOM series - tests convert on demand")
    print(f"  {group}: {nslices} DICOM slices, deliberately not converted")


if __name__ == "__main__":
    shapes()
    many()
    dicom()
    print("  done - open them from the project list.")
