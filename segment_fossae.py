#!/usr/bin/env python
"""
Cranial fossa volumes from the floor map alone.

ONE IDEA: measure the height of the skull floor over the whole axial footprint,
then read the two boundaries off that map. The anterior fossa floor is a high
shelf, the middle fossae are the basins behind it, and the petrous ridges are the
crests between those basins and the posterior fossa. Nothing else is consulted -
no landmark model decides where a boundary goes.

THE ONE THING THAT IS NOT FROM THE FLOOR: the direction the map is measured along.
"Height" needs an axis, and it has to be the head's own vertical, not the
scanner's, or a tilted-back head measures a tilted floor and every wall leans.
That axis comes from the glabella-to-torcula ceiling when it is available, and
falls back to the scanner vertical when it is not. Boundaries are extruded along
it, so the walls stay perpendicular to the base of the head.

Steps, in order:
  1. ICV = union of the brain_structures labels AND the `total` task's brain
     mask, cut at the foramen magnum.
  2. Frame: up (above), left-right from the skull's mirror symmetry, forward =
     up x left-right, pointing away from the cerebellum.
  3. Floor map: per 4mm lane and 1.5mm front-back bin, the height of the lowest
     intracranial voxel.
  4. Boundaries: per position, score how step-like (anterior) and how crest-like
     (posterior) the profile is, then fit one curve per boundary jointly across
     lanes - maximise total score, penalising both movement and change of
     direction between neighbouring lanes.
  5. Label every voxel by which side of the curves its column falls on, which is
     the vertical extrusion.

    python segment_fossae.py --group fossa
    python segment_fossae.py --group fossa --case CASE_A
    python segment_fossae.py --group fossa --case CASE_A CASE_B CASE_C

Writes <case>/fossae_simple.nii.gz / .seg.nrrd / .stats.json / _curves.npz
and the floor-map picture _map.png (--no-map to skip).
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from segment_structures import (seg_dir_for, find_source_nifti,
                                multilabel_to_segnrrd, group_dir)
from totalsegmentator.map_to_binary import class_map

# ---- from the plane-based pipeline this file replaces ------------------------
BRAIN_TASK = "brain_structures"
LABEL_VALUES = {"anterior_fossa": 1, "middle_fossa": 2, "posterior_fossa": 3}
COLORS = {
    "anterior_fossa": "0.550 0.250 0.850",
    "middle_fossa": "0.150 0.550 0.900",
    "posterior_fossa": "0.150 0.800 0.300",
}
MANUAL_LABELS = ("g", "o", "n", "s", "op",
                 "ACP(L)", "ACP(R)", "ZMF(L)", "ZMF(R)", "PR(L)", "PR(R)")
SEED_GROUPS = {
    "anterior_fossa": ("frontal_lobe",),
    "middle_fossa": ("temporal_lobe", "parietal_lobe", "insular_cortex",
                     "caudate_nucleus", "lentiform_nucleus", "internal_capsule",
                     "thalamus", "septum_pellucidum", "central_sulcus", "ventricle"),
    "posterior_fossa": ("cerebellum", "occipital_lobe"),
}
_T0 = time.perf_counter()

def log(msg, indent=0):
    print(f"[{time.perf_counter() - _T0:6.1f}s] {'  ' * indent}{msg}", flush=True)

def load_bool(path, ref_shape):
    d = np.asarray(nib.load(str(path)).dataobj) > 0
    if d.shape != ref_shape:
        raise ValueError(f"grid mismatch: {path} has {d.shape}, CT has {ref_shape}")
    return d

def unit(v):
    return v / np.linalg.norm(v)

def world_z(affine, shape):
    nx, ny, nz = shape
    ii = np.arange(nx, dtype=np.float32)[:, None, None]
    jj = np.arange(ny, dtype=np.float32)[None, :, None]
    kk = np.arange(nz, dtype=np.float32)[None, None, :]
    return (affine[2, 0] * ii + affine[2, 1] * jj + affine[2, 2] * kk
            + affine[2, 3]).astype(np.float32)

def mask_world_coords(mask, affine, max_pts=20000):
    """World (RAS) coordinates of a subsample of a mask's voxels, shape (N, 3)."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    idx = idx[:: max(1, len(idx) // max_pts)]
    return idx @ affine[:3, :3].T + affine[:3, 3]

def extreme_point(coords, direction):
    """World coordinate of the point furthest along `direction`."""
    return coords[np.argmax(coords @ direction)]

def _norm_name(s):
    return "".join(c for c in s.upper() if c.isalpha())

def load_manual_landmarks(group, case):
    """All labeled points (RAS) from <group>/points/*.csv for this case, or {}.
    Used only when exactly one file matches the case name."""
    pdir = group_dir(group) / "points"
    if not pdir.is_dir():
        return {}
    want = _norm_name(case)
    matches = []
    for f in sorted(pdir.glob("*.csv")):
        stem = _norm_name(f.stem.replace("Points", ""))
        if stem == want:
            matches.append(f)
    if len(matches) != 1:
        if len(matches) > 1:
            log(f"{len(matches)} landmark files match {case} (multiple timepoints?) - "
                f"cannot pair safely; not using manual landmarks", 2)
        return {}
    import csv as _csv
    pts = {}
    with open(matches[0], newline="") as f:
        for row in _csv.reader(f, delimiter="\t"):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) == 1 and "," in row[0]:
                row = row[0].split(",")
            if len(row) >= 4 and row[0].strip().lower() != "label":
                try:
                    x, y, z = (float(v) for v in row[1:4])
                except ValueError:
                    continue
                pts[row[0].strip()] = np.array([-x, -y, z])   # LPS -> RAS
    log(f"manual landmarks: {matches[0].name} ({len(pts)} points)", 2)
    return {k: v for k, v in pts.items() if k in MANUAL_LABELS}

def predicted_landmarks(group, case, seg_out, bone_ctx):
    """Predicted glabella/clinoids/opisthion from label_cranial_bones.py, running the
    model if needed. Returns {} on failure."""
    stats_p = seg_out / "cranial_bones.stats.json"
    if not stats_p.exists():
        try:
            import label_cranial_bones as lcb
            if "model" not in bone_ctx:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                log(f"loading cranial landmark model on {device}", 2)
                bone_ctx["model"] = torch.jit.load(str(lcb.MODEL_PATH),
                                                   map_location=device)
                bone_ctx["model"].eval()
                bone_ctx["device"] = device
            lcb.process_case(group, case, bone_ctx["model"], bone_ctx["device"])
        except Exception as e:
            log(f"WARNING: cranial landmark model failed: {e}", 2)
            return {}
    if not stats_p.exists():
        return {}
    with open(stats_p) as f:
        marks = json.load(f)["landmarks_ras"]
    return {k: np.asarray(v, dtype=float) for k, v in marks.items()}

def run_task_for_case(group, case, device, task, extra=()):
    """Run segment_structures.py for ONE study folder, into the group's output tree."""
    subprocess.run(
        [sys.executable, "segment_structures.py", str(group_dir(group) / case),
         "--group-name", group, "--task", task,
         "--skip-planning", "--device", device, *extra],
        check=True,
    )

def ensure_group_segmented(group, cases, device):
    def has_output(case):
        d = seg_dir_for(group) / case / BRAIN_TASK
        return d.is_dir() and any(d.glob("*.nii.gz"))

    missing = [c for c in cases if not has_output(c)]
    if missing:
        log(f"{len(missing)} case(s) lack {BRAIN_TASK} output - converting + segmenting "
            f"each (the slow part)")
        for c in missing:
            try:
                run_task_for_case(group, c, device, BRAIN_TASK)
            except subprocess.CalledProcessError as e:
                log(f"WARNING: segmentation failed for {c}: {e}")
    ready = [c for c in cases if has_output(c)]
    skipped = [c for c in cases if c not in ready]
    if skipped:
        log(f"WARNING: no {BRAIN_TASK} output for: {', '.join(skipped)}")
        log(f"  (likely needs manual series selection: python segment_structures.py "
            f"\"{group}\" --task {BRAIN_TASK})")
    return ready

def ensure_brain_floor(group, case, seg_out, ref_shape, device):
    """World-Z floor mask at the lowest voxel of total/brain (foramen magnum level)."""
    brain_p = seg_out / "total" / "brain.nii.gz"
    if not brain_p.exists():
        log("total/brain.nii.gz missing - running total (brain ROI only) for this case", 2)
        run_task_for_case(group, case, device, "total", ("--roi-subset", "brain"))
    if not brain_p.exists():
        log("WARNING: no total/brain mask - foramen magnum floor NOT applied", 2)
        return None
    brain_img = nib.load(str(brain_p))
    brain = np.asarray(brain_img.dataobj) > 0
    if brain.shape != ref_shape or not brain.any():
        log("WARNING: total/brain unusable - foramen magnum floor NOT applied", 2)
        return None
    wz = world_z(brain_img.affine, ref_shape)
    zmin = float(wz[brain].min())
    return wz >= (zmin - 1e-3)

def discover_cases(group):
    for root in (seg_dir_for(group),
                 group_dir(group) / f"converted_nifti_{group}"):
        if root.is_dir():
            cases = sorted(d.name for d in root.iterdir() if d.is_dir())
            if cases:
                return cases
    gdir = group_dir(group)
    if not gdir.is_dir():
        sys.exit(f"No such group folder: {group}")
    skip = (f"converted_nifti_{group}", f"total_segmentor_results_{group}", "points")
    return sorted(d.name for d in gdir.iterdir() if d.is_dir() and d.name not in skip)

def clean_seed(mask, voxel_ml, min_ml=0.5, min_frac=0.05):
    """Drop stray specks from a TS structure mask before it may seed - TS scatters
    small misclassified clumps at lobe boundaries and each would anchor a
    wrong-compartment island. A lobe legitimately has one component per hemisphere,
    so keep components >= min_frac of the largest AND >= min_ml."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    keep_vox = max(min_ml / voxel_ml, min_frac * sizes.max())
    keep = np.flatnonzero(sizes >= keep_vox) + 1
    return np.isin(lab, keep)

def build_anchors(manual, predicted, cereb, sinus, affine, icv_centroid):
    """Resolve every geometric anchor, preferring manual landmarks, then predictions,
    then mask-derived estimates. Returns (anchors dict, sources dict) or (None, why)."""
    a, src = {}, {}

    # clinoids - required
    if "ACP(L)" in manual and "ACP(R)" in manual:
        a["acp_l"], a["acp_r"] = manual["ACP(L)"], manual["ACP(R)"]
        src["clinoids"] = "manual"
    elif "clinoid_left" in predicted and "clinoid_right" in predicted:
        a["acp_l"], a["acp_r"] = predicted["clinoid_left"], predicted["clinoid_right"]
        src["clinoids"] = "predicted"
    else:
        return None, "no clinoid landmarks (manual or predicted)"

    # glabella-ish forward reference - required for orientation
    if "g" in manual:
        a["g"] = manual["g"]
        src["g"] = "manual"
    elif "glabella" in predicted:
        a["g"] = predicted["glabella"]
        src["g"] = "predicted"
    else:
        return None, "no glabella landmark for orientation"

    # Directions come from the image frame, not from landmarks: in RAS the +x axis IS
    # patient-left by definition, and predicted landmarks are too fragile to define a
    # frame (10mm-apart clinoids skew every plane). Landmarks supply POSITIONS only,
    # projected to the midline where the anatomy is midline.
    lr = np.array([1.0, 0.0, 0.0])
    mid_x = float(icv_centroid[0])
    for key in list(a.keys()):
        if key in ("g",):
            a[key] = np.array([mid_x, a[key][1], a[key][2]])

    # ceiling front anchor: upper orbital rim (ZMF midpoint) else glabella, projected
    # to the midline (its y/z level is what defines the ceiling; x must be midline)
    if "ZMF(L)" in manual and "ZMF(R)" in manual:
        fr = (manual["ZMF(L)"] + manual["ZMF(R)"]) / 2.0
        src["ceiling_front"] = "manual ZMF midpoint"
    else:
        fr = a["g"]
        src["ceiling_front"] = f"{src['g']} glabella (no ZMF)"
    a["front"] = np.array([mid_x, fr[1], fr[2]])

    # ceiling back anchor: internal occipital protuberance ~ torcula. The torcula is
    # the confluence of sinuses ON the tentorium, i.e. adjacent to the cerebellum -
    # restricting candidates to sinus voxels near the cerebellum keeps the superior
    # sagittal sinus (which runs high up the occiput and would drag the ceiling up the
    # vault) out of consideration.
    pts = mask_world_coords(sinus, affine)
    if pts is not None:
        near_mid = pts[np.abs(pts[:, 0] - mid_x) < 15.0]         # within 15mm of midline
        if len(near_mid):
            back_dir = np.array([0.0, -1.0, 0.0])                # posterior in RAS
            a["back"] = extreme_point(near_mid, back_dir)
            a["back"] = np.array([mid_x, a["back"][1], a["back"][2]])
            src["ceiling_back"] = "torcula (venous_sinuses posterior midline, near cerebellum)"
    if "back" not in a:
        pts = mask_world_coords(cereb, affine)
        if pts is None:
            return None, "no venous_sinuses or cerebellum mask for the ceiling anchor"
        a["back"] = extreme_point(pts, unit(np.array([0.0, -1.0, 1.0])))
        a["back"] = np.array([mid_x, a["back"][1], a["back"][2]])
        src["ceiling_back"] = "cerebellum posterior-superior edge (no venous_sinuses)"

    return a, src

def build_planes(anchors, cereb, affine, icv_centroid):
    """From resolved anchors: (ceiling, divider1, divider2) as (normal, point) pairs,
    plus a note on the petrous source. Normal orientations: ceiling normal points at
    the vault ("up"); divider normals point anterior."""
    lr = np.array([1.0, 0.0, 0.0])                        # patient left-right (RAS)
    fb = unit(anchors["back"] - anchors["front"])         # front -> back along ceiling
    ceil_n = unit(np.cross(lr, fb))
    if ceil_n @ (icv_centroid - anchors["front"]) < 0:    # point it at the vault
        ceil_n = -ceil_n

    # divider through the clinoid MIDPOINT (position is trustworthy even when the
    # individual predicted points are not), oriented by the frame, ⊥ ceiling
    clin_mid = (anchors["acp_l"] + anchors["acp_r"]) / 2.0
    n1 = unit(np.cross(lr, ceil_n))
    if n1 @ (anchors["g"] - clin_mid) < 0:
        n1 = -n1

    # petrous axis: manual PR pair if present, else the cerebellum's anterior-superior
    # extreme per side (where the tentorium attaches to the petrous ridge)
    if "PR(L)" in anchors and "PR(R)" in anchors:
        p_l, p_r = anchors["PR(L)"], anchors["PR(R)"]
        pet_src = "manual PR landmarks"
    else:
        pts = mask_world_coords(cereb, affine)
        if pts is None:
            return None, None, None, "no cerebellum mask for petrous estimate"
        fwd = unit(np.cross(ceil_n, lr))
        if fwd @ (anchors["g"] - clin_mid) < 0:
            fwd = -fwd
        score_dir = unit(fwd + ceil_n)                    # anterior-superior
        # Sample the ridge at its mid-lateral body (15-50mm off midline), NOT at the
        # extremes: the unrestricted anterior-superior extreme lands at the petrous
        # APEX (medial), which drags the plane forward and inflates the posterior
        # compartment. The published plane follows the superior petrous border.
        side = pts[:, 0] - clin_mid[0]
        left = pts[(side > 15.0) & (side < 50.0)]
        right = pts[(side < -15.0) & (side > -50.0)]
        if not len(left) or not len(right):
            return None, None, None, "cerebellum mask too small to split by side"
        p_l = extreme_point(left, score_dir)
        p_r = extreme_point(right, score_dir)
        pet_src = "cerebellum anterior-superior edge, mid-lateral (no PR landmarks)"

    pet_mid = (p_l + p_r) / 2.0
    ax_pet = unit(p_l - p_r)
    n2 = unit(np.cross(ax_pet, ceil_n))
    if n2 @ (anchors["g"] - pet_mid) < 0:
        n2 = -n2

    return (ceil_n, anchors["front"]), (n1, clin_mid), (n2, pet_mid), pet_src

def _cache_key(bdir, brain_p):
    """Fingerprint of everything the cached arrays depend on: the TotalSegmentator
    outputs themselves plus the config that turns them into seeds. Deliberately does
    NOT include the geometry code - planes, guards, extrusion and boundary mode can
    all change without invalidating the cache."""
    import hashlib
    files = {}
    for f in sorted(bdir.glob("*.nii.gz")):
        st = f.stat()
        files[f.name] = [int(st.st_mtime), st.st_size]
    if brain_p.exists():
        st = brain_p.stat()
        files["total/brain.nii.gz"] = [int(st.st_mtime), st.st_size]
    cfg = repr(sorted((g, tuple(n)) for g, n in SEED_GROUPS.items())) + "|clean0.5/0.05"
    return {"inputs_version": 1,
            "config": hashlib.md5(cfg.encode()).hexdigest(),
            "files": files}

def load_case_inputs(group, case, seg_out, bdir, ref_shape, zooms, voxel_ml, device,
                     use_cache=True, need_voronoi=True):
    """The expensive, geometry-independent part of a case: the structure masks, the
    hole-filled and floored ICV, the cleaned seeds, and the nearest-seed (Voronoi)
    assignment. Cached in <case>/fossae_inputs.npz, so repeated runs while iterating
    on the geometry skip straight to the partition."""
    cache_p = seg_out / "fossae_inputs.npz"
    vor_p = seg_out / "fossae_voronoi.npz"
    key_p = seg_out / "fossae_inputs.key.json"
    brain_p = seg_out / "total" / "brain.nii.gz"
    key = _cache_key(bdir, brain_p)
    if use_cache and cache_p.exists() and key_p.exists():
        try:
            with open(key_p) as f:
                fresh = json.load(f) == key
            if fresh and (vor_p.exists() or not need_voronoi):
                z = np.load(cache_p)
                out = {k: z[k] for k in ("icv", "seeds", "cereb", "sinus", "stem")}
                if need_voronoi:
                    out["voronoi"] = np.load(vor_p)["voronoi"]
                log(f"inputs: cache hit ({cache_p.name}"
                    f"{' + ' + vor_p.name if need_voronoi else ''})", 2)
                return out
        except Exception as e:
            log(f"inputs: cache unreadable ({e}) - recomputing", 2)

    seed_of = {n: g for g, names in SEED_GROUPS.items() for n in names}
    icv = np.zeros(ref_shape, dtype=bool)
    seeds = np.zeros(ref_shape, dtype=np.uint8)
    cereb = None
    stem_mask = np.zeros(ref_shape, dtype=bool)
    sinus = np.zeros(ref_shape, dtype=bool)
    for f in sorted(bdir.glob("*.nii.gz")):
        stem = f.name[:-7]
        m = load_bool(f, ref_shape)
        icv |= m
        if stem == "cerebellum":
            cereb = m
        elif stem == "venous_sinuses":
            sinus = m
        elif stem == "brainstem":
            stem_mask = m
        grp = seed_of.get(stem)
        if grp:
            seeds[clean_seed(m, voxel_ml)] = LABEL_VALUES[grp]
    if cereb is None or not cereb.any():
        return None
    icv = ndimage.binary_fill_holes(icv)

    floor = ensure_brain_floor(group, case, seg_out, ref_shape, device)
    if floor is not None:
        below = int((icv & ~floor).sum()) * voxel_ml
        icv &= floor
        seeds[~icv] = 0
        log(f"cut {below:.1f} mL below the foramen-magnum floor", 2)

    # torcula candidates: sinus voxels within 12mm of the cerebellum (this EDT is
    # otherwise recomputed on every run inside build_anchors)
    if sinus.any():
        dist_to_cereb = ndimage.distance_transform_edt(~cereb, sampling=zooms)
        sinus = sinus & (dist_to_cereb <= 12.0)

    out = {"icv": icv.astype(np.uint8), "seeds": seeds,
           "cereb": cereb.astype(np.uint8), "sinus": sinus.astype(np.uint8),
           "stem": stem_mask.astype(np.uint8)}
    # The nearest-seed (Voronoi) assignment is by far the most expensive step and is
    # only used by the curved-boundary partition in THIS script; the floor-profile
    # script never reads it. It therefore lives in its own cache file, so callers
    # that do not need it neither compute nor load it.
    voronoi = None
    if need_voronoi:
        log("computing nearest-seed assignment (Voronoi EDT)", 2)
        _, (ix, iy, iz) = ndimage.distance_transform_edt(
            seeds == 0, sampling=zooms, return_indices=True)
        voronoi = seeds[ix, iy, iz]
        del ix, iy, iz
        out["voronoi"] = voronoi
    else:
        log("skipping the nearest-seed assignment (not needed by this caller)", 2)

    if use_cache:
        try:
            np.savez_compressed(cache_p, **{k: v for k, v in out.items()
                                            if k != "voronoi"})
            if voronoi is not None:
                np.savez_compressed(vor_p, voronoi=voronoi)
            with open(key_p, "w") as f:
                json.dump(key, f)
            log(f"inputs: cached ({cache_p.stat().st_size/1e6:.0f} MB"
                f"{f' + {vor_p.stat().st_size/1e6:.0f} MB voronoi' if voronoi is not None else ''})", 2)
        except Exception as e:
            log(f"inputs: cache write failed ({e})", 2)
    return out

def _validate():
    valid = set(class_map[BRAIN_TASK].values())
    bad = [n for names in SEED_GROUPS.values() for n in names if n not in valid]
    if bad:
        raise RuntimeError(f"not in TotalSegmentator {BRAIN_TASK} class_map: {bad}")


_validate()

# --------------------------------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from segment_structures import (seg_dir_for, find_source_nifti,
                                multilabel_to_segnrrd, group_dir)

LANE_MM = 4.0        # lane width across the head
FBIN_MM = 1.5        # bin size front-to-back
MIN_COL_MM = 8.0     # a bin needs a column this tall to count as floor
FLANK_MM = 9.0       # half-width of the step / crest templates
JUMP_MM = 6.0        # a lane-to-lane move this big costs one unit of score
BEND_MM = 4.0        # a change of direction this big costs one unit of score
ANT_SCOPE = (0.45, 0.97)   # anterior search: fraction of the head's own extent
POST_SCOPE_MM = 30.0       # posterior search: +-this about the cerebellum's front


# --------------------------------------------------------------- the floor map
def floor_map(lane, fwd_mm, height, l_edges, f0, nf):
    """Height of the lowest intracranial voxel per (lane, front-back) cell."""
    li = np.searchsorted(l_edges, lane, side="right") - 1
    fi = np.clip(((fwd_mm - f0) / FBIN_MM).astype(np.int32), 0, nf - 1)
    nl = len(l_edges) - 1
    ok = (li >= 0) & (li < nl)
    bid = li[ok].astype(np.int64) * nf + fi[ok]
    h = height[ok]
    order = np.lexsort((h, bid))
    b, hs = bid[order], h[order]
    first = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    last = np.r_[first[1:] - 1, len(b) - 1]
    lo = np.full(nl * nf, np.nan, np.float32)
    hi = np.full(nl * nf, np.nan, np.float32)
    lo[b[first]] = hs[first]
    hi[b[first]] = hs[last]
    lo = lo.reshape(nl, nf)
    lo[(hi.reshape(nl, nf) - lo) < MIN_COL_MM] = np.nan   # rim, not a real floor
    return lo


def occupancy(lane, fwd_mm, l_edges, f0, nf):
    """Which cells contain any intracranial voxel."""
    li = np.searchsorted(l_edges, lane, side="right") - 1
    fi = np.clip(((fwd_mm - f0) / FBIN_MM).astype(np.int32), 0, nf - 1)
    nl = len(l_edges) - 1
    ok = (li >= 0) & (li < nl)
    occ = np.zeros((nl, nf), bool)
    occ[li[ok], fi[ok]] = True
    return occ


def smooth(prof):
    """Fill interior gaps and lightly smooth one lane's profile."""
    good = np.isfinite(prof)
    if good.sum() < 5:
        return np.full_like(prof, np.nan)
    i = np.arange(len(prof))
    out = prof.copy()
    out[~good] = np.interp(i[~good], i[good], prof[good])
    out = ndimage.gaussian_filter1d(out, 1.5)
    out[: np.argmax(good)] = np.nan
    out[len(good) - np.argmax(good[::-1]):] = np.nan
    return out


def _mean(prof, a_mm, b_mm):
    """Mean of prof over [f+a, f+b] at every f, NaN-aware."""
    a, b = int(round(a_mm / FBIN_MM)), int(round(b_mm / FBIN_MM))
    v = np.nan_to_num(prof)
    w = np.isfinite(prof).astype(float)
    cv, cw = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(w)]
    n = len(prof)
    i = np.arange(n)
    lo, hi = np.clip(i + a, 0, n), np.clip(i + b, 0, n)
    num, den = cv[hi] - cv[lo], cw[hi] - cw[lo]
    out = np.full(n, np.nan)
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


def score(prof, kind):
    """How step-like (anterior) or crest-like (posterior) each position is.

    Dimensionless by construction, so nothing depends on an absolute size and the
    same settings suit a small skull and a large one. Laterally the petrous crest
    runs out and the boundary becomes a plain step down onto the temporal fossa
    floor, so the posterior score takes whichever of the two is stronger.

    The anterior boundary is read as the steepest part of the climb off the middle
    fossa floor, not as the foot of it and not as the top: a difference of means
    across the position peaks exactly at the point of fastest rise."""
    F = FLANK_MM
    if kind == "step":                       # steepest rise going forward
        return _mean(prof, 0, F) - _mean(prof, -F, 0)
    core = _mean(prof, -F / 3, F / 3)
    crest = (core - _mean(prof, -F, -F / 2)) + 0.5 * (core - _mean(prof, F / 2, F))
    stepdown = _mean(prof, -F, 0) - _mean(prof, 0, F)
    return np.maximum(crest, stepdown)


def level_response(prof, level, span):
    """How close the floor is to one height, on a rising edge.

    The map is coloured by height on a single scale, so the edge the eye follows -
    where one colour meets the next - is one iso-level of the whole surface. Taking
    the level per lane instead gives every lane its own threshold, which traces no
    edge at all: lanes with no shelf still produce a crossing, and the curve wanders
    off the colour boundary. So the level is measured once, over the whole map.

    Reading the crossing rather than the steepest point is what makes it immune to
    the shape of the climb. A knee, a ramp and a clean step all cross the same level
    in the same place; only a slope template cares which of the three it is. Only
    rising crossings count, so the descent behind the ridge is not a match."""
    filled = np.nan_to_num(np.asarray(prof, float), nan=level - span)
    r = 1.0 - np.abs(filled - level) / max(span, 1e-6)
    r[np.gradient(filled) <= 0] = 0.0
    return np.maximum(r, 0.0)


def contour_level(fmap, f, inwin, behind):
    """The one height that separates basin from shelf, over the whole map.

    Measured only over ground the boundary can lie on - inside the window and ahead
    of the posterior boundary - so the posterior fossa floor, which is far deeper
    and already spoken for, cannot drag the level down."""
    vals = []
    for i in range(fmap.shape[0]):
        w = inwin if behind is None else (inwin & (f > behind[i]))
        v = fmap[i][w]
        vals.append(v[np.isfinite(v)])
    v = np.concatenate(vals) if vals else np.array([])
    if v.size < 20:
        return None, None
    lo, hi = np.percentile(v, 15), np.percentile(v, 85)
    return 0.5 * (lo + hi), 0.5 * max(hi - lo, 1e-6)


def score_map(fmap, f0, window, kind, occ, behind=None):
    """Score every lane at every position, normalised to unit noise scale.

    Negative means "does not look like the feature", which is an absence of
    evidence rather than evidence for somewhere else, so it is clipped to zero -
    otherwise lanes past the end of the ridge chase the least-bad position and
    kink the curve. Positions outside the anatomical window, or outside that
    lane's own brain, are forbidden outright."""
    nl, nf = fmap.shape
    f = f0 + np.arange(nf) * FBIN_MM
    inwin = (f >= window[0]) & (f <= window[1])
    level, span = ((None, None) if kind != "level"
                   else contour_level(fmap, f, inwin, behind))
    if kind == "level" and level is None:
        kind = "step"
    R = np.zeros((nl, nf))
    for i in range(nl):
        p = smooth(fmap[i])
        if not np.isfinite(p).any():
            continue
        R[i] = np.nan_to_num(level_response(p, level, span) if kind == "level"
                             else score(p, kind))
    scale = np.median(np.abs(R[R != 0])) if (R != 0).any() else 1.0
    R = np.maximum(R / max(scale, 1e-6), 0.0)
    R[:, ~inwin] = -np.inf
    # Keep each lane's boundary inside its own brain - but only for lanes with a
    # real cross-section. At the outermost lanes the head narrows to a sliver, and
    # forcing the curve into it drags both boundaries together there, pinching the
    # middle fossa to nothing at the very edge. Thin lanes are left unconstrained so
    # the smoothness term simply carries the curve through them.
    width = occ.sum(axis=1)
    for i in range(nl):
        cols = np.flatnonzero(occ[i])
        if len(cols) > 4 and width[i] >= 0.35 * width.max():
            bad = np.ones(nf, bool)
            bad[cols[0] + 2: cols[-1] - 1] = False
            R[i][bad] = -np.inf
    return R, f, float(scale)


def fit(R, f):
    """One position per lane: maximise total score, charging both the size of each
    lane-to-lane move and any change of direction.

    Charging only the size leaves corners free - a run of small steps that turns
    sharply costs the same as a straight run - so the curve comes out kinked where
    the profile is ambiguous. A lane with no admissible position is made neutral
    rather than being allowed to collapse the whole fit."""
    R = np.array(R, float, copy=True)
    R[~np.isfinite(R).any(axis=1)] = 0.0
    nl, nf = R.shape
    fj = f.astype(np.float32)
    jump = (((fj[None, :] - fj[:, None]) / JUMP_MM) ** 2).astype(np.float32)
    bend = (((fj[:, None, None] - 2 * fj[None, :, None] + fj[None, None, :])
             / BEND_MM) ** 2).astype(np.float32)
    if nl < 3:
        k = np.argmax(R, axis=1)
        return f[k], R[np.arange(nl), k]
    V = R[0][:, None] + R[1][None, :] - jump
    back = np.zeros((nl, nf, nf), np.int16)
    for i in range(2, nl):
        cand = V[:, :, None] - bend
        back[i] = np.argmax(cand, axis=0).astype(np.int16)
        V = np.take_along_axis(cand, back[i][None].astype(np.intp), axis=0)[0] \
            + R[i][None, :] - jump
    path = np.zeros(nl, np.int64)
    path[-2], path[-1] = np.unravel_index(int(np.argmax(V)), V.shape)
    for i in range(nl - 1, 1, -1):
        path[i - 2] = back[i][path[i - 1], path[i]]
    return f[path], np.array([R[i, path[i]] for i in range(nl)])


# ------------------------------------------------------------------- the frame
def mirror_axis(icv, affine, up, max_pts=400000):
    """The skull's left-right axis, from the angle whose mirror image overlaps the
    mask best. Returns (unit vector, point on the midline, angle, overlap)."""
    xw = np.array([1.0, 0.0, 0.0])
    e1 = unit(xw - (xw @ up) * up)
    e2 = unit(np.cross(up, e1))
    idx = np.argwhere(icv)
    if len(idx) > max_pts:
        idx = idx[:: len(idx) // max_pts + 1]
    P = idx @ affine[:3, :3].T + affine[:3, 3]
    c = P.mean(0)
    a, b, hh = (P - c) @ e1, (P - c) @ e2, (P - c) @ up
    BIN = 2.5
    best = (-1.0, 0.0, 0.0)
    for ang in np.arange(-45, 45.01, 1.0):
        t = np.radians(ang)
        u, v = a * np.cos(t) + b * np.sin(t), -a * np.sin(t) + b * np.cos(t)
        med = np.median(u)
        iu = np.round((u - med) / BIN).astype(int)
        iv = np.round(v / BIN).astype(int)
        ih = np.round(hh / BIN).astype(int)
        iu, iv, ih = iu - iu.min(), iv - iv.min(), ih - ih.min()
        G = np.zeros((iu.max() + 1, iv.max() + 1, ih.max() + 1), bool)
        G[iu, iv, ih] = True
        dice = (G & G[::-1]).sum() / max(G.sum(), 1)
        if dice > best[0]:
            best = (dice, ang, med)
    dice, ang, med = best
    t = np.radians(ang)
    lr = unit(np.cos(t) * e1 + np.sin(t) * e2)
    if lr[0] < 0:
        lr, med = -lr, -med
    return lr, c + med * lr, ang, dice


def head_frame(group, case, seg_out, icv, cereb, sinus, affine, centroid,
               use_landmarks=True, bone_ctx=None):
    """up (perpendicular to the base of the head), left-right, forward."""
    bone_ctx = {} if bone_ctx is None else bone_ctx
    up, how = np.array([0.0, 0.0, 1.0]), "scanner vertical"
    if use_landmarks:
        try:
            anchors, _ = build_anchors(load_manual_landmarks(group, case),
                                       predicted_landmarks(group, case, seg_out,
                                                           bone_ctx),
                                       cereb, sinus, affine, centroid)
            if anchors is not None:
                ceiling, _, _, _ = build_planes(anchors, cereb, affine, centroid)
                if ceiling is not None:
                    up, how = unit(np.asarray(ceiling[0], float)), "glabella-torcula"
        except Exception as e:
            log(f"landmark ceiling unavailable ({type(e).__name__}) - "
                f"using the scanner vertical", 2)
    lr, mid, yaw, dice = mirror_axis(icv, affine, up)
    up = unit(up - (up @ lr) * lr)                 # keep the frame orthonormal
    fwd = unit(np.cross(up, lr))
    cc = affine[:3, :3] @ np.argwhere(cereb).mean(0) + affine[:3, 3]
    if fwd @ (centroid - cc) < 0:                  # forward = away from cerebellum
        fwd = -fwd
    tilt = np.degrees(np.arccos(np.clip(abs(up @ np.array([0, 0, 1.0])), 0, 1)))
    log(f"frame: up from {how} ({tilt:.0f} deg off the scanner vertical), "
        f"yaw {yaw:+.0f} deg (mirror overlap {dice:.2f})", 2)
    return up, lr, mid, fwd, how, tilt, yaw, dice


# ------------------------------------------------------------------ the picture
def out_stem(case, name):
    """Output filenames carry the case, e.g. CASE_A_fossae_simple.seg.nrrd.

    The folder already names the case, but the file does not travel with its folder:
    loaded into Slicer it becomes a node called whatever the file was called, and in
    a zip of fifty cases every one of them would be fossae_simple.seg.nrrd. Spaces
    become underscores so the name survives a shell without quoting."""
    return "_".join(str(case).split()) + "_" + name


def out_path(seg_out, case, name, suffix):
    """Where to WRITE one output. Always the case-prefixed name."""
    return seg_out / f"{out_stem(case, name)}{suffix}"


def find_out(seg_out, case, name, suffix):
    """Where to READ one output: the case-prefixed name, or the bare one left by a
    run from before outputs were named per case."""
    new = out_path(seg_out, case, name, suffix)
    old = seg_out / f"{name}{suffix}"
    return new if new.exists() or not old.exists() else old


def slicer_view(ct_path, vol, affine):
    """The sagittal slice as Slicer would show it: CT in greyscale, segmentation
    painted over it, on the scanner's own voxel grid.

    Both volumes are put into canonical RAS first, so the first axis is left-right
    and the picture comes out with anterior to the right and superior up whatever
    orientation the scan was stored in. The slice shown is the one carrying the most
    middle fossa, so all three compartments are in it.

    Note this is a voxel plane, not a plane of the head's frame. If the head is
    pitched, correctly perpendicular walls WILL look tilted here - that is what they
    look like in Slicer too, and it is the reason this view alone cannot tell a
    leaning segmentation from a leaning head."""
    ct = nib.as_closest_canonical(nib.load(str(ct_path)))
    lb = nib.as_closest_canonical(nib.Nifti1Image(vol, affine))
    L = np.asarray(lb.dataobj)
    counts = (L == LABEL_VALUES["middle_fossa"]).sum(axis=(1, 2))
    i = int(np.argmax(counts)) if counts.any() else L.shape[0] // 2
    img = np.asarray(ct.dataobj[i, :, :], dtype=np.float32)
    zy, zz = ct.header.get_zooms()[1:3]
    return img, L[i], [0.0, img.shape[0] * zy, 0.0, img.shape[1] * zz], i


def render_map(case, fmap, f0, lc, ant, post, ant_str, post_str, head, out,
               lanes=(-40, -20, 0, 20, 40), sag=None):
    """The floor map with the two fitted boundaries drawn on it, plus the
    front-back profile along a few lanes - which is the 1D view the boundaries are
    actually read from. Written next to the segmentation it describes.

    With out=None the figure is returned instead of saved, so a caller can put it
    on a page of a multi-page PDF and keep the curves, axes and text as vectors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.colors import ListedColormap
    fb = f0 + np.arange(fmap.shape[1]) * FBIN_MM
    ncol = 2 if sag is None else 3
    fig = plt.figure(figsize=(8 * ncol, 7.5))
    ax = fig.add_subplot(1, ncol, 1)
    im = ax.imshow(fmap.T, origin="lower", aspect="equal",
                   extent=[lc[0], lc[-1], fb[0], fb[-1]], cmap="viridis")
    ax.plot(lc, ant, color="magenta", lw=2.5, label="anterior boundary")
    ax.plot(lc, post, color="cyan", lw=2.5, label="posterior boundary")
    ax.set_xlabel("mm right of midline (patient right)")
    ax.set_ylabel("mm forward")
    ax.set_title(f"{case}: floor height over the axial footprint (1:1)\n"
                 "light = high shelf, dark = deep basin", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=ax, label="floor height (mm)", shrink=0.8)

    ax = fig.add_subplot(1, ncol, 2)
    for want in lanes:
        i = int(np.argmin(np.abs(lc - want)))
        prof = smooth(fmap[i])
        ax.plot(fb, prof, lw=1.4, label=f"lane {lc[i]:+.0f} mm")
        col = ax.get_lines()[-1].get_color()
        pz = np.nan_to_num(prof, nan=0.0)
        ax.plot([ant[i]], [np.interp(ant[i], fb, pz)], "v", color=col, ms=7)
        ax.plot([post[i]], [np.interp(post[i], fb, pz)], "^", color=col, ms=7)
    ax.set_xlabel("mm forward")
    ax.set_ylabel("floor height (mm)")
    ax.set_title("front-back profile per lane\n"
                 "v = anterior boundary, ^ = posterior boundary", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if sag is not None:
        img, lab, extent, islice = sag
        ax = fig.add_subplot(1, ncol, 3)
        ax.imshow(img.T, origin="lower", aspect="equal", cmap="gray",
                  vmin=-200, vmax=1300, extent=extent)          # bone window
        cmap = ListedColormap([tuple(float(c) for c in COLORS[g].split())
                               for g in LABEL_VALUES])
        ax.imshow(np.ma.masked_equal(lab.T, 0) - 1, origin="lower", aspect="equal",
                  vmin=0, vmax=len(LABEL_VALUES) - 1, cmap=cmap, alpha=0.45,
                  extent=extent, interpolation="nearest")
        ax.set_xlabel("mm (posterior to anterior)")
        ax.set_ylabel("mm (inferior to superior)")
        ax.set_title(f"sagittal slice {islice} as Slicer shows it\n"
                     "voxel plane, so a pitched head tilts the walls here",
                     fontsize=10)

    def _ev(v):
        return f"{np.nanmedian(v):.1f}x" if np.isfinite(v).any() else "traced by hand"
    fig.suptitle(f"{head}   |   evidence: anterior {_ev(ant_str)}, "
                 f"posterior {_ev(post_str)}",
                 fontsize=10)
    fig.tight_layout()
    if out is None:                 # the caller wants the figure, e.g. for a PDF page
        return fig
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return None


# -------------------------------------------------------------------- per case
def head_frame_adjusted(up_adjust, up, lr, mid, fwd, how, tilt, yaw, dice):
    """Tilt the extrusion axis by hand, keeping the frame orthonormal.

    `up` is the direction the boundaries are extruded along, so it IS the direction
    the walls run. It is derived from the glabella-torcula ceiling, and when that
    lands wrong every wall leans by the same amount - which is visible in a scan and
    impossible to fix by moving the boundaries. Two angles are enough to correct it:
    pitch tips the axis front-to-back, roll tips it side to side, both in the head's
    own frame so they mean the same thing whatever the patient's position.

    Rotating one vector alone would leave the frame skewed, so each rotation is
    applied to the pair it moves and the third axis is left alone."""
    if not up_adjust:
        return up, lr, mid, fwd, how, tilt, yaw, dice
    pitch, roll = (float(a) for a in up_adjust)
    if pitch:
        t = np.radians(pitch)
        up, fwd = (np.cos(t) * up + np.sin(t) * fwd,
                   -np.sin(t) * up + np.cos(t) * fwd)
    if roll:
        t = np.radians(roll)
        up, lr = (np.cos(t) * up + np.sin(t) * lr,
                  -np.sin(t) * up + np.cos(t) * lr)
    up, lr = unit(up), unit(lr)
    fwd = unit(np.cross(up, lr))
    tilt = np.degrees(np.arccos(np.clip(abs(up @ np.array([0, 0, 1.0])), 0, 1)))
    how = f"{how} adjusted by hand (pitch {pitch:+.0f} deg, roll {roll:+.0f} deg)"
    log(f"extrusion axis tilted by hand: pitch {pitch:+.0f} deg, roll {roll:+.0f} deg "
        f"-> {tilt:.0f} deg off the scanner vertical", 2)
    return up, lr, mid, fwd, how, tilt, yaw, dice


def process_case(group, case, name, device, use_landmarks=True, bone_ctx=None,
                 make_map=True, traced=None, up_adjust=None,
                 posterior="ridge"):
    seg_out = seg_dir_for(group) / case
    ct_path = find_source_nifti(group, case)
    if not ct_path or not Path(ct_path).exists():
        log(f"SKIP {case}: no converted CT")
        return None
    log(f"case: {group} / {case}")
    img = nib.load(str(ct_path))
    affine, shape = img.affine, img.shape
    zooms = img.header.get_zooms()[:3]
    vml = float(np.prod(zooms)) / 1000.0

    inputs = load_case_inputs(group, case, seg_out, seg_out / "brain_structures",
                              shape, zooms, vml, device, True, need_voronoi=False)
    if inputs is None:
        log(f"SKIP {case}: no cerebellum mask")
        return None
    icv, cereb, sinus = (inputs["icv"] > 0, inputs["cereb"] > 0,
                         inputs["sinus"] > 0)
    bp = seg_out / "total" / "brain.nii.gz"
    bt = None
    if bp.exists():                                # foramen-magnum cut
        bt = np.asarray(nib.load(str(bp)).dataobj) > 0
        if bt.shape == icv.shape and bt.any():
            # union of the two brain masks, not just the structure labels. They fail
            # in different places and neither contains the other: the total task's
            # brain holds tens of mL the structure labels miss, and vice versa. The
            # structure part is still clipped to a dilated total/brain so a stray
            # label cannot reach far outside the brain, but total/brain itself is
            # added whole - it is already cut at the foramen magnum.
            icv = (icv & ndimage.binary_dilation(bt, iterations=3)) | bt
        else:
            bt = None
    icv_ml = int(icv.sum()) * vml
    centroid = affine[:3, :3] @ np.argwhere(icv).mean(0) + affine[:3, 3]
    log(f"ICV {icv_ml:.0f} mL   voxel {zooms[0]:.2f} x {zooms[1]:.2f} x "
        f"{zooms[2]:.2f} mm", 2)

    up, lr, mid, fwd, how, tilt, yaw, dice = head_frame_adjusted(up_adjust, *head_frame(
        group, case, seg_out, icv, cereb, sinus, affine, centroid, use_landmarks,
        bone_ctx if bone_ctx is not None else {}))

    idx = np.argwhere(icv)
    W = idx @ affine[:3, :3].T + affine[:3, 3]
    h = (W - mid) @ up
    lane = (W - mid) @ lr
    fv = (W - mid) @ fwd
    del W
    l_edges = np.arange(lane.min() - LANE_MM, lane.max() + 2 * LANE_MM, LANE_MM)
    lc = (l_edges[:-1] + l_edges[1:]) / 2.0
    f0 = float(fv.min())
    nf = int((fv.max() - fv.min()) / FBIN_MM) + 2
    fmap = floor_map(lane, fv, h, l_edges, f0, nf)
    occ = occupancy(lane, fv, l_edges, f0, nf)
    # How much relief the map actually has to work with. A base with real ridges
    # gives a large front-back swing per lane; a smooth bowl gives a small one, and
    # then neither template has anything to lock onto whatever the fit does.
    _sw = [np.nanmax(fmap[i]) - np.nanmin(fmap[i]) for i in range(fmap.shape[0])
           if np.isfinite(fmap[i]).sum() > 5]
    log(f"floor map {fmap.shape[0]} lanes x {nf} bins, "
        f"{100.0 * np.isfinite(fmap).sum() / occ.sum():.0f}% of footprint has floor, "
        f"relief {np.median(_sw):.0f} mm per lane (median)", 2)

    lo, hi = np.percentile(fv, 1), np.percentile(fv, 99)
    ant_win = (lo + ANT_SCOPE[0] * (hi - lo), lo + ANT_SCOPE[1] * (hi - lo))
    # The petrous ridge runs along the cerebellum's front-upper edge, so centre the
    # posterior search there. Sampled at the cerebellum's mid-lateral body, not at
    # its extremes: the very front-most point is the petrous APEX near the midline,
    # which sits well ahead of the ridge and drags the window forward.
    _cw = np.argwhere(cereb) @ affine[:3, :3].T + affine[:3, 3] - mid
    _cl, _cf = _cw @ lr, _cw @ fwd
    _band = (np.abs(_cl) > 15.0) & (np.abs(_cl) < 50.0)
    if _band.sum() < 200:
        _band = np.ones(len(_cl), bool)
    cf = float(np.percentile(_cf[_band], 95))    # the cerebellum's front edge
    post_win = (cf - POST_SCOPE_MM, cf + POST_SCOPE_MM)

    # "ridge" puts the boundary on the crest of the petrous bone, which is where the
    # tentorium attaches and how the posterior fossa is conventionally bounded.
    # "basin" puts it where the floor drops to posterior-fossa depth, one slope
    # further back - the edge the eye follows on the map. The two differ by the width
    # of the petrous bone's posterior slope, about a centimetre.
    Rp, f_ax, sp = score_map(fmap, f0, post_win,
                             "level" if posterior == "basin" else "crest", occ)
    # The foramen magnum lies inside the posterior fossa, and at the midline the
    # boundary is the dorsum sellae - well in front of the basion. So over the
    # foramen's own width the boundary cannot fall behind its anterior rim. Without
    # this the crest template is free to take any ridge in its window, including the
    # occipital one behind the foramen, which is a different bone entirely.
    if bt is not None:
        bw = np.argwhere(bt) @ affine[:3, :3].T + affine[:3, 3] - mid
        bh = bw @ up
        ring = bw[bh <= bh.min() + 5.0]            # the cut face = the foramen
        if len(ring) > 50:
            rl, rf = ring @ lr, ring @ fwd
            fm_front, fm_half = float(np.percentile(rf, 95)), float(np.percentile(np.abs(rl), 95))
            near = np.abs(lc) <= fm_half
            Rp[np.ix_(near, f_ax <= fm_front)] = -np.inf
            log(f"foramen magnum: front rim {fm_front:+.0f} mm, half-width "
                f"{fm_half:.0f} mm - boundary held in front of it over "
                f"{int(near.sum())} lanes", 2)
    post, post_str = fit(Rp, f_ax)
    Ra, _, sa = score_map(fmap, f0, ant_win, "level", occ, behind=post)
    for i in range(Ra.shape[0]):                   # anterior stays in front
        blocked = f_ax <= post[i]
        if not blocked.all():
            Ra[i][blocked] = -np.inf
    ant, ant_str = fit(Ra, f_ax)
    log(f"evidence: anterior {np.nanmedian(ant_str):.1f}x, "
        f"posterior {np.nanmedian(post_str):.1f}x the noise scale", 2)

    # A boundary traced by hand replaces the fitted one, and nothing else changes -
    # the same labelling, the same extrusion, the same outputs. Points are (lane mm,
    # forward mm) and are resampled onto this case's own lane grid, so a trace made
    # on the map lines up with the lanes whatever the map's resolution was.
    for key, arr in (("ant", None), ("post", None)):
        pts = (traced or {}).get(key)
        if not pts:
            continue
        q = np.asarray(sorted(pts, key=lambda t: t[0]), float)
        vals = np.interp(lc, q[:, 0], q[:, 1])
        if key == "ant":
            ant, ant_str = vals, np.full(len(lc), np.nan)
        else:
            post, post_str = vals, np.full(len(lc), np.nan)
        log(f"{key}: replaced by a traced boundary ({len(q)} points)", 2)
    # A curve that spans almost nothing, or that sits on the edge of its own search
    # window, was not detected - it was cornered. Worth seeing without opening the map.
    for nm, cv, wn in (("anterior", ant, ant_win), ("posterior", post, post_win)):
        log(f"{nm}: {cv.min():+.0f} to {cv.max():+.0f} mm forward "
            f"(spans {cv.max() - cv.min():.0f} mm) in window "
            f"{wn[0]:+.0f} to {wn[1]:+.0f}", 2)

    # ---- label by column: this IS the vertical extrusion ---------------------
    # lane and fv are both measured perpendicular to `up`, so neither changes as
    # you travel up a column. Labelling by (lane, fv) therefore carries the floor
    # boundary straight up, perpendicular to the base of the head, and no rule
    # runs afterwards that could bend it.
    lcl = np.clip(lane, lc[0], lc[-1])
    a_at = PchipInterpolator(lc, ant, extrapolate=False)(lcl)
    p_at = PchipInterpolator(lc, post, extrapolate=False)(lcl)
    lab = np.full(len(idx), LABEL_VALUES["middle_fossa"], np.uint8)
    lab[fv < p_at] = LABEL_VALUES["posterior_fossa"]
    lab[(fv >= p_at) & (fv > a_at)] = LABEL_VALUES["anterior_fossa"]
    vol = np.zeros(shape, np.uint8)
    vol[tuple(idx.T)] = lab

    out_img = nib.Nifti1Image(vol, affine)
    nib.save(out_img, str(out_path(seg_out, case, name, ".nii.gz")))
    multilabel_to_segnrrd(out_img, {v: k for k, v in LABEL_VALUES.items()},
                          out_path(seg_out, case, name, ".seg.nrrd"), colors=COLORS)
    np.savez_compressed(out_path(seg_out, case, name, "_curves.npz"),
                        l_centers=lc, ant=ant,
                        post=post, ant_str=ant_str, post_str=post_str, fmap=fmap,
                        f0=np.float64(f0), tilt=np.float64(tilt),
                        yaw=np.float64(yaw), dice=np.float64(dice),
                        how=np.array(how))
    if make_map:
        mp = out_path(seg_out, case, name, "_map.png")
        try:
            sag = slicer_view(ct_path, vol, affine)
            render_map(case, fmap, f0, lc, ant, post, ant_str, post_str,
                       f"{how} up-axis, {tilt:.0f} deg off the scanner vertical", mp,
                       sag=sag)
            log(f"map -> {mp}", 2)
        except Exception as e:
            log(f"WARNING: map not written ({type(e).__name__}: {e})", 2)

    frac = {g: 100.0 * float((vol == v).sum()) * vml / icv_ml
            for g, v in LABEL_VALUES.items()}
    stats = {"case": case, "ct": os.path.basename(str(ct_path)),
             "method": "boundaries read off the skull-floor map; extruded along the "
                       "head's own vertical",
             "icv_ml": round(icv_ml, 1),
             "posterior_boundary": posterior,
             "up_axis": how, "up_tilt_vs_scanner_deg": round(tilt, 1),
             "up_adjust_pitch_roll_deg": list(up_adjust) if up_adjust else None,
             "midsagittal_yaw_deg": round(yaw, 1), "mirror_overlap": round(dice, 3),
             "evidence_anterior_x": round(float(np.nanmedian(ant_str)), 2)
             if np.isfinite(ant_str).any() else None,
             "evidence_posterior_x": round(float(np.nanmedian(post_str)), 2)
             if np.isfinite(post_str).any() else None,
             "traced": sorted((traced or {}).keys()) or None,
             "compartments": {g: {"ml": round(float((vol == v).sum()) * vml, 1),
                                  "percent_of_icv": round(frac[g], 1)}
                              for g, v in LABEL_VALUES.items()}}
    if frac["anterior_fossa"] > 30 or frac["anterior_fossa"] < 5 \
            or frac["middle_fossa"] < 10 or frac["posterior_fossa"] < 20:
        stats["warning"] = "implausible compartment fractions"
        log("WARNING: implausible compartment fractions - check the map", 2)
    with open(out_path(seg_out, case, name, ".stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    log("volumes (% ICV): " + "  ".join(
        f"{g.split('_')[0]} {frac[g]:.1f}%" for g in LABEL_VALUES), 2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="fossa")
    ap.add_argument("--case", nargs="+", default=None,
                    help="one or more case folder names; omit for the whole group")
    ap.add_argument("--name", default="fossae_simple")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"])
    ap.add_argument("--no-landmarks", action="store_true",
                    help="use the scanner vertical instead of the glabella-torcula "
                         "ceiling (the head is then assumed upright)")
    ap.add_argument("--posterior", choices=["ridge", "basin"], default="ridge",
                    help="what the middle/posterior boundary is. ridge (default): the "
                         "crest of the petrous bone, where the tentorium attaches. "
                         "basin: where the floor drops to posterior-fossa depth, about "
                         "a centimetre further back. ridge matches published posterior "
                         "fossa volumetry; basin follows the visible edge on the map "
                         "but its position depends on how steep the ridge is, which "
                         "changes with age")
    ap.add_argument("--up-adjust", type=float, nargs=2, metavar=("PITCH", "ROLL"),
                    default=None,
                    help="tilt the extrusion axis by hand, in degrees, in the head's "
                         "own frame: pitch tips it front-to-back, roll side to side")
    ap.add_argument("--ignore-corrections", action="store_true",
                    help="refit from the floor map even for cases corrected in the "
                         "review site, ignoring their <name>_traced.json")
    ap.add_argument("--no-map", action="store_true",
                    help="skip the floor-map picture (it is written by default as "
                         "<name>_map.png in the case's results folder)")
    args = ap.parse_args()

    cases = args.case if args.case else discover_cases(args.group)
    cases = ensure_group_segmented(args.group, cases, args.device)
    # One shared context so the cranial landmark model is loaded once per run,
    # not once per case - otherwise a group run pays that load for every case.
    bone_ctx = {}
    for c in cases:
        try:
            # A case corrected in the review site keeps its correction on a re-run.
            # Without this the corrections are silently overwritten by the automatic
            # result the next time the group is processed, which is the one way a
            # piece of someone's work could quietly disappear here.
            edits = {}
            ep = find_out(seg_dir_for(args.group) / c, c, args.name,
                          "_traced.json")
            if ep.exists() and not args.ignore_corrections:
                edits = json.loads(ep.read_text())
                log(f"{c}: applying saved corrections "
                    f"({', '.join(sorted(edits))}) - --ignore-corrections to refit")
            traced = {k: edits[k] for k in ("ant", "post") if k in edits}
            process_case(args.group, c, args.name, args.device,
                         use_landmarks=not args.no_landmarks, bone_ctx=bone_ctx,
                         make_map=not args.no_map, posterior=args.posterior,
                         traced=traced or None,
                         up_adjust=args.up_adjust or edits.get("up_adjust"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"ERROR {c}: {e}")


if __name__ == "__main__":
    main()
