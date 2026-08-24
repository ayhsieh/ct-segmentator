#!/usr/bin/env python
"""
Label the five cranial bone plates (left/right frontal, left/right parietal, occipital)
and detect four cranial-base landmarks (glabella, left/right clinoid process, opisthion)
on a head CT, using the cuMIP CranialCTProcessing model:

    Joint Cranial Bone Labeling and Landmark Detection in Pediatric CT Images using
    Context Encoding.  https://github.com/cuMIP/CranialCTProcessing  (MIT license)
    Model weights: cranial_ct_processing/Model.dat (TorchScript, included).

Outputs per case, alongside the other pipeline results:
    cranial_bones.seg.nrrd / .nii.gz   five bone plates as named segments
    cranial_landmarks.mrk.json         Slicer markups file with the four landmarks
    cranial_bones.stats.json           per-bone volumes + landmark coordinates (RAS)

IMPORTANT - naming is geometric, confirm once visually. The upstream repo does not
document which output channel is which bone/landmark, so this script names them by
anatomy-independent geometry (in the patient's own axes, computed from the CT affine):
    bones:     occipital = most posterior centroid; frontals = two most anterior
               (left/right by patient X); parietals = the remaining two
    landmarks: glabella = most anterior; opisthion = most posterior;
               clinoids = the middle two (left/right by patient X)
Open one case's cranial_bones.seg.nrrd in Slicer and check the names look right; if a
name is wrong the geometry rule needs adjusting, not the model.

Model caveats: trained on ages 0-2 (roughly half craniosynostosis patients), inference
at 96x96x96 (labels are resampled back to the native grid, masked to the bone mask).
Older children are out of distribution - inspect before trusting.

The preprocessing (adaptive bone threshold, convex-hull masking, resample+normalize)
is vendored from the upstream DataProcessing.py, minus its vtk/torchio dependencies.

Usage:
    python label_cranial_bones.py --group fossa                 # all cases
    python label_cranial_bones.py --group fossa --case CASE_A
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from scipy.spatial import ConvexHull, Delaunay
from skimage import measure

from segment_structures import seg_dir_for, find_source_nifti, multilabel_to_segnrrd

MODEL_PATH = Path(__file__).parent / "cranial_ct_processing" / "Model.dat"

BONE_LABELS = ["frontal_left", "frontal_right", "parietal_left", "parietal_right",
               "occipital"]
LANDMARK_NAMES = ["glabella", "clinoid_left", "clinoid_right", "opisthion"]

_T0 = time.perf_counter()


def log(msg, indent=0):
    print(f"[{time.perf_counter() - _T0:6.1f}s] {'  ' * indent}{msg}", flush=True)


# ------------------------------------------------------------------ preprocessing
# Vendored from cuMIP/CranialCTProcessing DataProcessing.py (MIT), with the torchio
# rescale replaced by a plain min-max normalization and vtk removed.
def flood_fill_hull(image, max_voxels=2_000_000):
    """Binary convex hull of the nonzero voxels. For large volumes the hull indicator is
    evaluated on a strided subgrid and nearest-upsampled - the hull only trims
    outside-the-head air, so a few voxels of boundary error are irrelevant, while
    evaluating Delaunay membership per full-res voxel would take minutes on a 512^3 CT."""
    stride = int(np.ceil((image.size / max_voxels) ** (1 / 3)))
    small = image[::stride, ::stride, ::stride] if stride > 1 else image
    points = np.transpose(np.where(small))
    deln = Delaunay(points[ConvexHull(points).vertices])
    idx = np.stack(np.indices(small.shape), axis=-1)
    out_small = (deln.find_simplex(idx) + 1 > 0)
    if stride == 1:
        return out_small.astype(np.float64)
    out = np.repeat(np.repeat(np.repeat(out_small, stride, 0), stride, 1), stride, 2)
    return out[:image.shape[0], :image.shape[1], :image.shape[2]].astype(np.float64)


def create_head_mask(ct_image, hu_threshold=-200):
    m = (sitk.GetArrayFromImage(ct_image) > hu_threshold).astype(np.uint8)
    lab = measure.label(m)
    m = (lab == np.argmax(np.bincount(lab.flat)[1:]) + 1).astype(np.uint8)
    out = sitk.GetImageFromArray(m)
    out.CopyInformation(ct_image)
    return out


def create_bone_mask(ct_image, hu_min=100, hu_max=200):
    """Adaptive threshold [Dangi et al. 2017]: pick the HU cut in hu_min..hu_max that
    yields the fewest connected components, keep the largest one."""
    head_img = create_head_mask(ct_image)          # keep the image alive; a view into a
    head = sitk.GetArrayFromImage(head_img)        # temporary's buffer is a silent segfault
    arr = sitk.GetArrayFromImage(ct_image).copy()
    arr[head == 0] = 0
    best, best_n = hu_min, np.inf
    for t in range(hu_min, hu_max + 1, 10):
        n = measure.label(arr >= t).max()
        if n < best_n:
            best_n, best = n, t
    lab = measure.label(arr >= best)
    bone = (lab == np.argmax(np.bincount(lab.flat)[1:]) + 1).astype(np.uint8)
    out = sitk.GetImageFromArray(bone)
    out.CopyInformation(ct_image)
    return out


def resample_and_mask(ct_image, bone_image, size=96):
    convex = flood_fill_hull(sitk.GetArrayFromImage(bone_image))
    arr = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else arr * 0
    arr[convex == 0] = 0
    norm = sitk.GetImageFromArray(arr)
    norm.CopyInformation(ct_image)

    template = sitk.Image([size] * 3, sitk.sitkFloat32)
    template.SetOrigin(ct_image.GetOrigin())
    template.SetDirection(ct_image.GetDirection())
    template.SetSpacing(tuple(np.array(ct_image.GetSpacing())
                              * np.array(ct_image.GetSize()) / size))
    return sitk.Resample(norm, template, sitk.Transform(), sitk.sitkLinear)


# ------------------------------------------------------------------ postprocessing
def sitk_to_nib(image):
    """SimpleITK (LPS) -> nibabel image (RAS affine), preserving the voxel grid."""
    arr = sitk.GetArrayFromImage(image)          # [z, y, x]
    sp = np.array(image.GetSpacing())
    d = np.array(image.GetDirection()).reshape(3, 3)
    o = np.array(image.GetOrigin())
    aff_lps = np.eye(4)
    aff_lps[:3, :3] = d @ np.diag(sp)
    aff_lps[:3, 3] = o
    aff_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff_lps
    return nib.Nifti1Image(arr.transpose(2, 1, 0), aff_ras)


def patient_axes(image):
    """Unit vectors (anterior, left, superior) in LPS world space - from the CT frame,
    so 'anterior' etc. follow the patient regardless of head tilt in the scanner...
    but note the CT frame IS the patient frame per DICOM convention (LPS axes are
    defined anatomically), so these are constants in LPS."""
    ant = np.array([0.0, -1.0, 0.0])   # -P = anterior
    left = np.array([1.0, 0.0, 0.0])   # +L = patient left
    sup = np.array([0.0, 0.0, 1.0])    # +S = superior
    return ant, left, sup


def name_bones(labels_img):
    """Map the model's arbitrary label values -> anatomical names, by geometry."""
    arr = sitk.GetArrayFromImage(labels_img)
    values = sorted(int(v) for v in np.unique(arr) if v > 0)
    if len(values) != 5:
        log(f"WARNING: expected 5 bone labels, got {len(values)}: {values}", 1)
    ant, left, _ = patient_axes(labels_img)
    cents = {}
    for v in values:
        idx = np.argwhere(arr == v)                       # [z,y,x]
        phys = [labels_img.TransformIndexToPhysicalPoint(
            (int(x), int(y), int(z))) for z, y, x in idx[::max(1, len(idx)//500)]]
        cents[v] = np.mean(phys, axis=0)
    if len(values) < 3:
        return {v: f"bone_{v}" for v in values}
    by_ant = sorted(values, key=lambda v: float(cents[v] @ ant))
    occipital = by_ant[0]                                  # most posterior
    frontals = by_ant[-2:]                                 # two most anterior
    parietals = [v for v in values if v not in (occipital, *frontals)]
    mapping = {occipital: "occipital"}
    for pair, base in ((frontals, "frontal"), (parietals, "parietal")):
        pair = sorted(pair, key=lambda v: float(cents[v] @ left))
        if len(pair) == 2:
            mapping[pair[1]] = f"{base}_left"              # larger +L = left
            mapping[pair[0]] = f"{base}_right"
        else:
            for v in pair:
                mapping[v] = f"{base}_unpaired_{v}"
    return mapping


def name_landmarks(points_lps):
    """points_lps: list of 4 (x,y,z) LPS coords in heatmap-channel order -> names."""
    ant, left, _ = patient_axes(None)
    order = sorted(range(len(points_lps)), key=lambda i: -(points_lps[i] @ ant))
    names = {}
    if len(points_lps) == 4:
        names[order[0]] = "glabella"                       # most anterior
        names[order[-1]] = "opisthion"                     # most posterior
        mid = sorted(order[1:3], key=lambda i: points_lps[i] @ left)
        names[mid[1]] = "clinoid_left"
        names[mid[0]] = "clinoid_right"
    else:
        for i in range(len(points_lps)):
            names[i] = f"landmark_{i}"
    return names


def write_markups(points_lps, names, path):
    """Slicer .mrk.json, coordinates stored in LPS (Slicer converts on load)."""
    cps = []
    for i, p in enumerate(points_lps):
        cps.append({"id": str(i + 1), "label": names[i],
                    "position": [float(p[0]), float(p[1]), float(p[2])],
                    "orientation": [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]})
    doc = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json",
           "markups": [{"type": "Fiducial", "coordinateSystem": "LPS",
                        "coordinateUnits": "mm", "controlPoints": cps}]}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


# ------------------------------------------------------------------ per-case
def process_case(group, case, model, device):
    seg_out = seg_dir_for(group) / case
    ct_path = find_source_nifti(group, case)
    if not ct_path or not Path(ct_path).exists():
        log(f"SKIP {case}: no converted CT NIfTI")
        return None
    seg_out.mkdir(parents=True, exist_ok=True)

    log(f"case: {group} / {case}")
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    log(f"CT {os.path.basename(str(ct_path))}  size={ct.GetSize()}  "
        f"spacing={tuple(round(s, 3) for s in ct.GetSpacing())} mm", 1)

    log("STEP 1/4  bone mask (adaptive threshold)", 1)
    bone = create_bone_mask(ct)
    log("STEP 2/4  convex-hull mask + resample to 96^3 + normalize", 1)
    net_in = resample_and_mask(ct, bone)

    log("STEP 3/4  inference", 1)
    x = torch.tensor(sitk.GetArrayFromImage(net_in), dtype=torch.float32,
                     device=device)[None, None]
    with torch.no_grad():
        seg_pred, heat_pred, _ = model(x)

    # bones: argmax at 96^3, resample to the native grid, mask to actual bone
    lab96 = np.argmax(seg_pred[0].cpu().numpy(), axis=0).astype(np.uint16)
    lab_img = sitk.GetImageFromArray(lab96)
    lab_img.CopyInformation(net_in)
    lab_native = sitk.Resample(lab_img, bone, sitk.Transform(),
                               sitk.sitkNearestNeighbor)
    arr = sitk.GetArrayFromImage(lab_native)
    arr[sitk.GetArrayViewFromImage(bone) == 0] = 0
    lab_native = sitk.GetImageFromArray(arr)
    lab_native.CopyInformation(bone)

    # landmarks: heatmap peaks -> physical points (honouring the direction matrix,
    # which upstream's origin+spacing*index skips)
    heat = heat_pred[0].cpu().numpy()
    pts = []
    for c in range(heat.shape[0]):
        z, y, xx = np.unravel_index(int(np.argmax(heat[c])), heat[c].shape)
        pts.append(np.array(net_in.TransformIndexToPhysicalPoint(
            (int(xx), int(y), int(z)))))

    log("STEP 4/4  name by geometry + write outputs", 1)
    bone_names = name_bones(lab_native)
    lm_names = name_landmarks(pts)

    nib_img = sitk_to_nib(lab_native)
    nib.save(nib_img, str(seg_out / "cranial_bones.nii.gz"))
    multilabel_to_segnrrd(nib_img, bone_names, seg_out / "cranial_bones.seg.nrrd")
    write_markups(pts, lm_names, seg_out / "cranial_landmarks.mrk.json")

    voxel_ml = float(np.prod(ct.GetSpacing())) / 1000.0
    arr = sitk.GetArrayViewFromImage(lab_native)
    stats = {"case": case, "ct": os.path.basename(str(ct_path)),
             "model": "cuMIP/CranialCTProcessing (naming: geometric, verify visually)",
             "bones": {n: {"label": int(v), "ml": round(float((arr == v).sum()) * voxel_ml, 1)}
                       for v, n in bone_names.items()},
             "landmarks_ras": {lm_names[i]: [round(-p[0], 1), round(-p[1], 1), round(p[2], 1)]
                               for i, p in enumerate(pts)}}
    with open(seg_out / "cranial_bones.stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    for n, e in stats["bones"].items():
        log(f"{n:<16} {e['ml']:>8.1f} mL", 2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="fossa")
    ap.add_argument("--case", default=None,
                    help="single case; omit to process every case with a converted CT")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"Model weights not found: {args.model}\n"
                 "Clone https://github.com/cuMIP/CranialCTProcessing and copy Model.dat "
                 "into cranial_ct_processing/, or pass --model.")

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"loading model on {device}")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    if args.case:
        cases = [args.case]
    else:
        nifti_root = Path(args.group) / f"converted_nifti_{args.group}"
        if not nifti_root.is_dir():
            sys.exit(f"No converted NIfTIs at {nifti_root} - run segment_structures.py "
                     f"(or segment_fossae.py) for this group first.")
        cases = sorted(d.name for d in nifti_root.iterdir() if d.is_dir())
    log(f"group: {args.group}  ({len(cases)} case(s))")

    results = []
    for case in cases:
        try:
            s = process_case(args.group, case, model, device)
        except Exception as e:
            log(f"ERROR {case}: {e}")
            s = None
        if s:
            results.append(s)

    print(f"\n{len(results)}/{len(cases)} case(s) labeled. Outputs: "
          f"{seg_dir_for(args.group)}\\<case>\\cranial_bones.* + cranial_landmarks.mrk.json")
    print("NOTE: bone/landmark names are assigned geometrically - confirm once in Slicer.")


if __name__ == "__main__":
    main()
