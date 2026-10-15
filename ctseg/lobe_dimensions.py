#!/usr/bin/env python
"""
Width, length and height of each cerebral lobe, left and right, in the head's own frame.

The four lobe masks from brain_structures (frontal, parietal, temporal, occipital) each
cover both hemispheres, so they are split at the midline. The midline is anatomical, not
the plane of best mirror symmetry: a symmetry fit drifts toward the larger side of a
lopsided head and hides part of the very asymmetry being measured. It is fitted through
two structures that sit on the true midline whatever the head's shape - the septum
pellucidum and the centre line of the brainstem - each given equal weight, so a large
septum does not outvote the brainstem's leverage further back.

  lr   the midline plane's normal, pointing to the patient's right (+x)
  fwd  the cerebrum's long axis within that plane, pointing to the frontal lobe; a
       nodding head would otherwise trade height for length
  up   perpendicular to both, pointing from the brainstem to the parietal lobe

Each extent runs from the 1st to the 99th percentile of the lobe's voxels along that
axis, so a stray voxel does not set a dimension. A lobe that touches the first or last
slice ran past the scan, and is marked partial.

Writes <results>/<case>/lobe_dimensions.stats.json.

    python -m ctseg.lobe_dimensions --group sample_ct --case SAMPLE1
"""
import argparse
import json
import sys

import nibabel as nib
import numpy as np

from ctseg.ct_paths import seg_dir_for

LOBES = ("frontal_lobe", "parietal_lobe", "temporal_lobe", "occipital_lobe")
MIN_SEPTUM_VOXELS = 30          # fewer than this and the septum gives no plane
SINUS_NEAR_MM = 10.0            # venous sinus voxels this close to the first fit are midline
PCT = (1, 99)


def load(p):
    img = nib.load(str(p))
    return np.asarray(img.dataobj) > 0, img.affine


def world(mask, aff):
    return np.argwhere(mask) @ aff[:3, :3].T + aff[:3, 3]


def unit(v):
    return v / np.linalg.norm(v)


def midline(d):
    """(point, normal) of the midline plane, or (None, why)."""
    sep, aff = load(d / "septum_pellucidum.nii.gz")
    stem, _ = load(d / "brainstem.nii.gz")
    if sep.sum() < MIN_SEPTUM_VOXELS:
        return None, "septum pellucidum too small to fit a midline"
    if not stem.any():
        return None, "no brainstem"
    S = world(sep, aff)
    # the brainstem as its centre line - one centroid per slice - so its bulk does not
    # widen the plane; it is there for leverage, far behind and below the septum
    ks = np.unique(np.nonzero(stem)[2])
    B = np.c_[[np.argwhere(stem[:, :, k]).mean(0) for k in ks], ks]
    B = B @ aff[:3, :3].T + aff[:3, 3]
    c, n, rms = fit_plane([S, B])
    # Septum and brainstem both sit in the middle of the head, so nothing holds the
    # plane's yaw at the back. The superior sagittal sinus runs the length of the
    # midline along the top, and is added once the first fit says where the midline is:
    # the venous_sinuses mask also holds the transverse sinuses, which run sideways, and
    # only voxels already near the plane belong to the midline.
    sin_p = d / "venous_sinuses.nii.gz"
    if sin_p.exists():
        sin, _ = load(sin_p)
        V = world(sin, aff) if sin.any() else np.zeros((0, 3))
        V = V[np.abs((V - c) @ n) < SINUS_NEAR_MM]
        if len(V) >= 50:
            c, n, rms = fit_plane([S, B, V])
    return (c, n, rms), None


def fit_plane(groups):
    """Least-squares plane through point groups that count equally, however many
    points each has. Returns (point, unit normal toward +x, weighted rms in mm)."""
    P = np.vstack(groups)
    w = np.concatenate([np.full(len(g), 1.0 / (len(groups) * len(g))) for g in groups])
    c = (P * w[:, None]).sum(0)
    _, _, vt = np.linalg.svd((P - c) * np.sqrt(w)[:, None], full_matrices=False)
    n = vt[2] if vt[2][0] >= 0 else -vt[2]
    return c, n, float(np.sqrt((w * ((P - c) @ n) ** 2).sum()))


def frame(d, c, n):
    cere = None
    for lobe in LOBES:
        m, aff = load(d / f"{lobe}.nii.gz")
        cere = m if cere is None else cere | m
    W = world(cere, aff)
    Q = (W - c) - np.outer((W - c) @ n, n)            # projected onto the midline plane
    _, _, vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
    fwd = unit(vt[0] - (vt[0] @ n) * n)
    front = world(load(d / "frontal_lobe.nii.gz")[0], aff).mean(0)
    if fwd @ (front - W.mean(0)) < 0:
        fwd = -fwd
    up = unit(np.cross(n, fwd))
    top = world(load(d / "parietal_lobe.nii.gz")[0], aff).mean(0)
    stem = world(load(d / "brainstem.nii.gz")[0], aff).mean(0)
    if up @ (top - stem) < 0:
        up = -up
    return fwd, up


def measure(group, case):
    d = seg_dir_for(group) / case / "brain_structures"
    if not all((d / f"{x}.nii.gz").exists()
               for x in LOBES + ("septum_pellucidum", "brainstem")):
        return {"error": "brain_structures masks missing"}
    got, why = midline(d)
    if got is None:
        return {"error": why}
    c, n, rms = got
    fwd, up = frame(d, c, n)
    out = {"midline": {"from": "septum pellucidum + brainstem centre line",
                       "fit_rms_mm": round(rms, 2)},
           "lobes": {}}
    for lobe in LOBES:
        m, aff = load(d / f"{lobe}.nii.gz")
        if not m.any():
            continue
        W = world(m, aff)
        side_of = (W - c) @ n
        rec = {"partial": bool(m[:, :, 0].any() or m[:, :, -1].any())}
        for side, sel in (("right", side_of > 0), ("left", side_of < 0)):
            if sel.sum() < 50:
                continue
            V = W[sel] - c
            span = lambda v: float(np.diff(np.percentile(v, PCT))[0])
            rec[side] = {"width_mm": round(span(np.abs(V @ n)), 1),
                         "length_mm": round(span(V @ fwd), 1),
                         "height_mm": round(span(V @ up), 1)}
        out["lobes"][lobe] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--case", nargs="+", default=None)
    ap.add_argument("--device", default=None, help="accepted and ignored: no model runs")
    a = ap.parse_args()
    root = seg_dir_for(a.group)
    cases = a.case or sorted(p.name for p in root.iterdir() if (p / "brain_structures").is_dir())
    for case in cases:
        res = measure(a.group, case)
        (root / case / "lobe_dimensions.stats.json").write_text(json.dumps(res, indent=2))
        print(case, res.get("error") or " ".join(
            f"{k[:-5]} R{v.get('right', {}).get('width_mm')}/L{v.get('left', {}).get('width_mm')}"
            for k, v in res["lobes"].items()), flush=True)


if __name__ == "__main__":
    sys.exit(main())
