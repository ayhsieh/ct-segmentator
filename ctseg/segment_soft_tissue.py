#!/usr/bin/env python
"""
Segment ALL soft tissue in a head/neck CT and write it as a single-segment 3D-Slicer
.seg.nrrd (plus a .nii.gz mask and a stats.json), matching the output conventions of
segment_structures.py.

There is no TotalSegmentator task that outputs "all soft tissue", so this combines
thresholding with TS anatomy:

  soft_tissue_threshold   TS `body` envelope  AND  (HU_MIN < HU < HU_MAX)
                          MINUS anatomic exclusions (see below)

The HU window removes air and cortical bone by attenuation alone, and drops dense
contrast/calcium (> HU_MAX). On its own it would still keep fatty marrow as islands inside
bone, a partial-volume rim around every cortical edge, and every fluid space. The anatomic
exclusions clean that up. What remains: skin, fat, muscle, glands, parenchyma.

TS supplies two things the CT numbers cannot: the `body` envelope, which is what separates
the patient from the scanner table and surrounding air (both inseparable by HU alone), and
the named structures subtracted below.

Anatomic exclusions, each a named group toggled with --exclude / --no-exclude:

  bone            whole TS bone masks, dilated 1 vox   (marrow + the partial-volume    [ON]
                                                        rim that survives the HU cut)
  csf             subarachnoid_space + ventricle              (CSF is fluid)          [ON]
  venous_sinuses  dural venous sinuses + falx/tentorium       (venous blood + dura)   [off]
  intracranial    all brain_structures labels + total/brain   (drop the whole ICV)    [off]
  cartilage       laryngeal + costal cartilage   (sits in the soft-tissue HU window)   [off]
  airways         pharynx, nasal cavity, paranasal sinuses, larynx_air, trachea       [off]
  orbital         globes, lenses, optic nerves                                        [off]
  glands          parotid + submandibular salivary glands                             [off]
  tongue          tongue                                                              [off]
  vessels         internal carotids/jugulars, aorta, brachiocephalic, ... (blood)     [off]
  spinal_cord     spinal cord / thecal sac                                            [off]
  aerodigestive   esophagus + soft/hard palate                                        [off]

PRESETS bundle an exclusion set with an inferior cut plane and an output name:

  --preset face   skin + subcutaneous fat + facial muscle, as an envelope: everything
                  inside the skin and outside the skull, minus every labelled deep
                  structure, floored at the inferior border of the mandible. The scalp
                  over the calvarium is included. The muscles of mastication (masseter,
                  temporalis, pterygoids) are KEPT - TotalSegmentator has no labels for
                  the mimetic muscles, so this cannot be built additively from TS output.
                  Caveat: orbital fat has no TS label, so it survives inside the orbit.

--floor {mandible,C2,C3} discards everything below the lowest voxel of that structure,
using per-voxel world Z from the full affine (so tilted acquisitions are handled).

OUTPUTS
    <name>.nii.gz / .seg.nrrd              the envelope, one segment
    <name>_composition.nii.gz / .seg.nrrd  skin / fat / muscle, three disjoint segments
                                           that sum exactly to the envelope
    <name>.stats.json                      volumes, per-group exclusion accounting

The skin layer is `--skin-mm` millimetres inward from the body surface (default 2.0),
restricted to voxels denser than `--fat-hu-max`. Its OUTER edge is the real skin/air
boundary. Its INNER edge is real wherever subcutaneous fat lies beneath - dermis is about
+10..+60 HU against fat at about -100 - but where skin sits directly on muscle or
cartilage (eyelids, nose, ears, lips) the two are isodense and --skin-mm caps the layer
instead. On thick slices a 1-2 mm skin layer is largely partial volume regardless, so skin
volumes are only comparable between cases acquired at the same slice thickness.
`--skin-mm 0` writes fat/muscle only. TotalSegmentator has no skin model: its `body` task
emits a skin.nii.gz, but that is morphology in postprocessing.py (erosion by VOXEL count,
so its thickness varies with slice spacing), not a segmentation.

Structures are named EXACTLY as in totalsegmentator.map_to_binary.class_map, and validated
against it at import - no substring matching, so a rename upstream is an error rather than a
silently missing mask.

Volumes are reported for EVERY group whose masks are already on disk, whether or not it is
subtracted, so the cost of each candidate exclusion is visible before enabling it.

Prerequisite TotalSegmentator tasks are produced by ctseg.segment_structures:
    total, craniofacial_structures, headneck_bones_vessels   (bone sources)
    body                                                     (patient envelope)
    brain_structures                                         (CSF / intracranial)
    head_glands_cavities                                     (airways, orbital, glands)
    head_muscles                                             (tongue)
The task an enabled exclusion group *requires* is run automatically via
segment_structures.py if missing; optional extra sources (e.g. the large `total`
vessels) are used only if that task was already run.

Usage (run from the ct-segmentator directory):
    python -m ctseg.segment_soft_tissue                       # defaults: see --group / --case
    python -m ctseg.segment_soft_tissue --group STUDY --case CASE_A
    python -m ctseg.segment_soft_tissue --exclude bone csf venous_sinuses vessels
    python -m ctseg.segment_soft_tissue --exclude intracranial      # extracranial soft tissue only
    python -m ctseg.segment_soft_tissue --no-exclude bone           # HU threshold only for bone
    python -m ctseg.segment_soft_tissue --exclude                   # pure HU threshold, no anatomy
    python -m ctseg.segment_soft_tissue --preset face               # facial soft-tissue envelope
    python -m ctseg.segment_soft_tissue --preset face --floor C2    # ... down to C2 instead
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

# Reuse the existing pipeline's path + NRRD conventions so output matches segment_structures.py
from ctseg.segment_structures import seg_dir_for, find_source_nifti, multilabel_to_segnrrd
from ctseg.ct_paths import case_dir_for
# TotalSegmentator's own label tables - every structure name below is checked against these
# at import, so a typo or an upstream rename is a loud error, not a silently dropped mask.
from totalsegmentator.map_to_binary import class_map

# --------------------------------------------------------------------------- config
SOFT_HU_MIN = -200.0   # below this = air  -> excluded from soft tissue
SOFT_HU_MAX = 300.0    # above this = cortical bone -> excluded from soft tissue

BODY_TASK = "body"
BRAIN_TASK = "brain_structures"
GLANDS_TASK = "head_glands_cavities"
MUSCLE_TASK = "head_muscles"
HEADNECK_TASK = "headneck_bones_vessels"
CRANIOFACIAL_TASK = "craniofacial_structures"

_VERTEBRAE = ([f"vertebrae_C{i}" for i in range(1, 8)] +
              [f"vertebrae_T{i}" for i in range(1, 13)] +
              [f"vertebrae_L{i}" for i in range(1, 6)] + ["vertebrae_S1"])
_RIBS = [f"rib_{s}_{i}" for s in ("left", "right") for i in range(1, 13)]

# Bone, by exact TotalSegmentator label name (used by the "bone" exclusion group).
BONE_NAMES = {
    "total": tuple(["skull", "sacrum", "sternum", "humerus_left", "humerus_right",
                    "scapula_left", "scapula_right", "clavicula_left", "clavicula_right"]
                   + _VERTEBRAE + _RIBS),
    CRANIOFACIAL_TASK: ("skull", "mandible", "teeth_lower", "teeth_upper"),
    HEADNECK_TASK: ("hyoid", "zygomatic_arch_left", "zygomatic_arch_right",
                    "styloid_process_left", "styloid_process_right"),
}

# Cartilage sits inside the soft-tissue HU window, so the threshold alone keeps it. Kept in
# its own table so removing it is a deliberate, separately toggleable choice.
CARTILAGE_NAMES = {
    "total": ("costal_cartilages",),
    HEADNECK_TASK: ("thyroid_cartilage", "cricoid_cartilage"),
}

# --------------------------------------------------------------------------- exclusions
# Structures that live inside the body envelope and pass the soft-tissue HU window, but are
# not soft tissue. Each group is a list of (task, names) sources, where names are EXACT
# TotalSegmentator label names; names=None means "every mask the task produced". Sources
# whose task was never run are skipped and reported, EXCEPT the tasks named in "requires",
# which are run on demand.
#   default  : subtracted unless --no-exclude turns it off
#   requires : task(s) segmented on demand when the group is enabled
EXCLUSION_GROUPS = {
    "bone": {
        # The HU window already drops cortical bone; this additionally removes the fatty
        # marrow / diploic space inside the bones (soft-tissue HU) and, via the 1-voxel
        # dilation, the partial-volume shell at the cortical edge.
        "sources": [(t, names) for t, names in BONE_NAMES.items()],
        "default": True,
        "dilate": 1,   # partial-volume rim at the cortical edge sits inside the HU window
        "requires": ["total"],
        "desc": "whole bones incl. marrow (skull, vertebrae, mandible, teeth, ribs, ...)",
    },
    "cartilage": {
        "sources": [(t, names) for t, names in CARTILAGE_NAMES.items()],
        "default": False,
        "requires": [],
        "desc": "laryngeal + costal cartilage (sits inside the soft-tissue HU window)",
    },
    "csf": {
        "sources": [(BRAIN_TASK, ("subarachnoid_space", "ventricle"))],
        "default": True,
        "requires": [BRAIN_TASK],
        "desc": "intracranial CSF (subarachnoid space + ventricles)",
    },
    "venous_sinuses": {
        "sources": [(BRAIN_TASK, ("venous_sinuses",))],
        "default": False,
        "requires": [BRAIN_TASK],
        "desc": "dural venous sinuses / falx / tentorium (venous blood + dura)",
    },
    "intracranial": {
        # every brain_structures label (parenchyma, CSF, sinuses) plus the coarse total/brain
        "sources": [(BRAIN_TASK, None), ("total", ("brain",))],
        "default": False,
        "requires": [BRAIN_TASK],
        "desc": "all intracranial contents (brain, CSF, dural sinuses)",
    },
    # NOTE: for a whole-body soft-tissue mask there is deliberately no "globes" exclusion -
    # TS labels the entire globe (sclera, choroid, lens), which is tissue. The "orbital"
    # group below exists for the facial-envelope preset, where orbital contents are not
    # part of "skin + subcutaneous fat + muscle".
    "orbital": {
        "sources": [(GLANDS_TASK, ("eye_left", "eye_right", "eye_lens_left", "eye_lens_right",
                                   "optic_nerve_left", "optic_nerve_right"))],
        "default": False,
        "requires": [GLANDS_TASK],
        "desc": "globes, lenses, optic nerves (orbital fat has no TS label and is kept)",
    },
    "glands": {
        "sources": [(GLANDS_TASK, ("parotid_gland_left", "parotid_gland_right",
                                   "submandibular_gland_left", "submandibular_gland_right"))],
        "default": False,
        "requires": [GLANDS_TASK],
        "desc": "parotid + submandibular salivary glands",
    },
    "tongue": {
        "sources": [(MUSCLE_TASK, ("tongue",))],
        "default": False,
        "requires": [MUSCLE_TASK],
        "desc": "tongue (deep oral structure, not part of the facial envelope)",
    },
    "airways": {
        # air-filled cavities; HU_MIN already drops the air itself, this also removes the
        # partial-volume rim and secretions/mucosal filling TS labels as cavity
        "sources": [(GLANDS_TASK, ("nasopharynx", "oropharynx", "hypopharynx",
                                   "nasal_cavity_left", "nasal_cavity_right",
                                   "auditory_canal_left", "auditory_canal_right")),
                    (HEADNECK_TASK, ("larynx_air",)),
                    (CRANIOFACIAL_TASK, ("sinus_maxillary", "sinus_frontal")),
                    ("total", ("trachea",))],
        "default": False,
        "requires": [GLANDS_TASK],
        "desc": "pharynx / nasal cavity / paranasal sinuses / larynx / trachea",
    },
    "vessels": {
        "sources": [(HEADNECK_TASK, ("internal_carotid_artery_left", "internal_carotid_artery_right",
                                     "internal_jugular_vein_left", "internal_jugular_vein_right")),
                    ("total", ("aorta", "brachiocephalic_trunk",
                               "brachiocephalic_vein_left", "brachiocephalic_vein_right",
                               "common_carotid_artery_left", "common_carotid_artery_right",
                               "subclavian_artery_left", "subclavian_artery_right",
                               "superior_vena_cava", "inferior_vena_cava", "pulmonary_vein"))],
        "default": False,
        "requires": [HEADNECK_TASK],
        "desc": "large vessel lumens (carotids, jugulars, aorta, ...) - luminal blood",
    },
    "spinal_cord": {
        "sources": [("total", ("spinal_cord",))],
        "default": False,
        "requires": ["total"],
        "desc": "spinal cord / thecal sac contents",
    },
    "aerodigestive": {
        "sources": [("total", ("esophagus",)),
                    (GLANDS_TASK, ("soft_palate", "hard_palate"))],
        "default": False,
        "requires": [],
        "desc": "esophagus + palate (mixed lumen/mucosa, usually keep)",
    },
}

DEFAULT_EXCLUSIONS = [g for g, spec in EXCLUSION_GROUPS.items() if spec["default"]]

# --------------------------------------------------------------------------- presets
# A preset is just a named (exclusions, floor, output-name) triple.
#
# "face": skin + subcutaneous fat + facial muscle, as an ENVELOPE. TotalSegmentator has no
# labels for the mimetic (facial expression) muscles - `head_muscles` covers only the
# muscles of mastication - so this cannot be built additively. Instead: everything inside
# the skin, outside the skull, that isn't a labelled deep structure. The muscles of
# mastication (masseter, temporalis, pterygoids) are KEPT as facial muscle.
PRESETS = {
    "face": {
        # "vessels" is deliberately omitted: the labelled vessels are intracranial or below
        # the mandible floor, so it would cost a whole extra TS run for ~no volume. Add it
        # explicitly with --exclude if you want it.
        "exclude": ["bone", "intracranial", "csf", "venous_sinuses", "airways",
                    "orbital", "glands", "tongue", "aerodigestive"],
        "floor": "mandible",
        "name": "facial_soft_tissue",
        "desc": "skin + subcutaneous fat + facial muscle, mandible floor, scalp included",
    },
}

# Inferior cut planes: keep only voxels at or above the lowest voxel of a reference mask.
# World-Z is computed per voxel from the full affine, so tilted acquisitions work too.
FLOORS = {
    "mandible": {"source": (CRANIOFACIAL_TASK, "mandible"),
                 "desc": "inferior border of the mandible"},
    "C2": {"source": ("total", "vertebrae_C2"),
           "desc": "inferior border of the C2 vertebral body"},
    "C3": {"source": ("total", "vertebrae_C3"),
           "desc": "inferior border of the C3 vertebral body"},
}


def _validate_names():
    """Every declared structure name must exist in TotalSegmentator's class_map for that task.
    Catches typos here and upstream renames on the next TS upgrade."""
    declared = [(t, n) for names in (BONE_NAMES, CARTILAGE_NAMES) for t, ns in names.items() for n in ns]
    for g, spec in EXCLUSION_GROUPS.items():
        for task, names in spec["sources"]:
            if names is not None:
                declared += [(task, n) for n in names]
    bad = []
    for task, name in declared:
        if task not in class_map:
            bad.append(f"{task}: unknown task")
        elif name not in set(class_map[task].values()):
            bad.append(f"{task}/{name}")
    if bad:
        raise RuntimeError("structure names not in TotalSegmentator class_map: "
                           + ", ".join(sorted(set(bad))))


_validate_names()


# --------------------------------------------------------------------------- progress
_T0 = time.perf_counter()


def log(msg, indent=0):
    """Timestamped progress line, flushed so it appears live during long mask unions."""
    print(f"[{time.perf_counter() - _T0:6.1f}s] {'  ' * indent}{msg}", flush=True)


# --------------------------------------------------------------------------- helpers
def load_bool(path, ref_shape):
    m = nib.load(str(path))
    d = np.asarray(m.dataobj) > 0
    if d.shape != ref_shape:
        raise ValueError(f"grid mismatch: {path} has {d.shape}, CT has {ref_shape}")
    return d


def ensure_task(group, seg_out, task, device):
    """Return the <task>/ structures dir, running the task via segment_structures if absent.

    The app runs every prerequisite as its own step first, so this only segments when
    the script is run by hand."""
    tdir = seg_out / task
    if tdir.is_dir() and any(tdir.glob("*.nii.gz")):
        return tdir
    print(f"[{task}] output not found - running segment_structures --task {task}")
    subprocess.run(
        [sys.executable, "-m", "ctseg.segment_structures",
         str(case_dir_for(group, seg_out.name)), "--group-name", group,
         "--task", task, "--skip-planning", "--device", device],
        check=True,
    )
    if not (tdir.is_dir() and any(tdir.glob("*.nii.gz"))):
        raise RuntimeError(f"{task} task did not produce masks in {tdir}")
    return tdir


def build_group(seg_out, sources, ref_shape, verbose=False, indent=2):
    """Union the masks named by a list of (task, exact-names-or-None) sources.
    Names are matched exactly against the exported <task>/<structure>.nii.gz filenames.
    A named structure the task should have produced but didn't is reported, not ignored.
    Returns (mask, files_used, unavailable)."""
    mask = np.zeros(ref_shape, dtype=bool)
    used, unavailable = [], []
    for task, names in sources:
        tdir = seg_out / task
        if not tdir.is_dir():
            unavailable.append(f"{task} (task not run)")
            if verbose:
                log(f"{task:<24} - not run, skipped", indent)
            continue
        files = (sorted(tdir.glob("*.nii.gz")) if names is None
                 else [tdir / f"{n}.nii.gz" for n in names])
        got, absent = 0, []
        for f in files:
            if not f.exists():
                # segment_structures.py only writes a file when the label is non-empty,
                # so this is normally just "structure outside the scan FOV".
                unavailable.append(f"{task}/{f.name[:-7]}")
                absent.append(f.name[:-7])
                continue
            mask |= load_bool(f, ref_shape)
            used.append(f"{task}/{f.name}")
            got += 1
        if verbose:
            note = ""
            if absent:
                shown = ", ".join(absent[:4]) + (", ..." if len(absent) > 4 else "")
                note = f"  ({len(absent)} absent: {shown})"
            log(f"{task:<24} {got:>3} mask(s){note}", indent)
    return mask, used, unavailable


def world_z(affine, shape):
    """Per-voxel world Z, from the full affine (handles tilted / oblique acquisitions)."""
    nx, ny, nz = shape
    ii = np.arange(nx, dtype=np.float32)[:, None, None]
    jj = np.arange(ny, dtype=np.float32)[None, :, None]
    kk = np.arange(nz, dtype=np.float32)[None, None, :]
    return (affine[2, 0] * ii + affine[2, 1] * jj + affine[2, 2] * kk + affine[2, 3]).astype(np.float32)


def build_floor(seg_out, floor, affine, ref_shape):
    """Boolean mask of everything at or above the lowest voxel of the floor's reference
    structure. Returns (mask, min_world_z) or (None, None) if the reference is unavailable."""
    task, name = FLOORS[floor]["source"]
    f = seg_out / task / f"{name}.nii.gz"
    if not f.exists():
        log(f"floor reference {task}/{name} not found - no inferior cut applied", 1)
        return None, None
    ref = load_bool(f, ref_shape)
    if not ref.any():
        log(f"floor reference {task}/{name} is empty - no inferior cut applied", 1)
        return None, None
    wz = world_z(affine, ref_shape)
    zmin = float(wz[ref].min())
    log(f"floor at {FLOORS[floor]['desc']}: world Z >= {zmin:.1f} mm", 1)
    return wz >= (zmin - 1e-3), zmin


def build_body(body_dir, ref_shape, verbose=False):
    """Union of body sub-structures (body_trunc + body_extremities), hole-filled to a solid envelope."""
    files = sorted(body_dir.glob("*.nii.gz"))
    if not files:
        raise RuntimeError(f"no body masks in {body_dir}")
    body = np.zeros(ref_shape, dtype=bool)
    for f in files:
        if verbose:
            log(f"union {f.name}", 2)
        body |= load_bool(f, ref_shape)
    if verbose:
        log("binary_fill_holes -> solid envelope", 2)
    body = ndimage.binary_fill_holes(body)
    return body


def write_outputs(mask, name, seg_out, affine, label="soft_tissue"):
    mask = mask.astype(np.uint8)
    img = nib.Nifti1Image(mask, affine)
    nib.save(img, str(seg_out / f"{name}.nii.gz"))
    multilabel_to_segnrrd(img, {1: label}, seg_out / f"{name}.seg.nrrd")


def build_composition(soft, body, hu, zooms, skin_mm, fat_hu_max):
    """Partition the soft-tissue envelope into skin / fat / muscle. The three are disjoint
    and their union is exactly `soft`.

      skin   : within `skin_mm` of the body surface AND denser than fat.
               The outer edge is the real skin/air boundary. The inner edge is real
               wherever subcutaneous fat lies beneath (dermis ~ +10..+60 HU vs fat ~ -100,
               so `fat_hu_max` finds it); where skin sits directly on muscle or cartilage
               there is no recoverable boundary and `skin_mm` caps the layer instead.
               Distance is in MILLIMETRES (sampling=zooms), not voxels, so the layer does
               not change thickness with slice spacing.
      fat    : everything at or below fat_hu_max (subcutaneous + deep/intermuscular).
      muscle : the rest - muscle and other lean soft tissue.

    Returns (skin, fat, muscle). skin_mm <= 0 disables the skin layer.
    """
    fat = soft & (hu <= fat_hu_max)
    lean = soft & ~fat
    if skin_mm <= 0:
        return np.zeros_like(soft), fat, lean
    # distance inward from the outside of the patient, in mm
    dist_mm = ndimage.distance_transform_edt(body, sampling=zooms)
    skin = lean & (dist_mm <= skin_mm)
    return skin, fat, lean & ~skin


# --------------------------------------------------------------------------- main
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="sample_ct")
    ap.add_argument("--case", default=None)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu", "mps"])
    ap.add_argument("--hu-min", type=float, default=SOFT_HU_MIN)
    ap.add_argument("--hu-max", type=float, default=SOFT_HU_MAX)
    ap.add_argument("--exclude", nargs="*", default=None, metavar="GROUP",
                    choices=list(EXCLUSION_GROUPS),
                    help="exclusion groups to subtract (replaces the defaults: "
                         + ", ".join(DEFAULT_EXCLUSIONS) + "). Choices: "
                         + ", ".join(EXCLUSION_GROUPS))
    ap.add_argument("--no-exclude", nargs="*", default=[], metavar="GROUP",
                    choices=list(EXCLUSION_GROUPS),
                    help="drop these groups from whatever set is active")
    ap.add_argument("--exclude-dilate", type=int, default=0, metavar="N",
                    help="dilate the exclusion mask by N voxels before subtracting, to also "
                         "clear the partial-volume rim (default 0)")
    ap.add_argument("--preset", choices=list(PRESETS), default=None,
                    help="named configuration. " + "; ".join(
                        f"{k}: {v['desc']}" for k, v in PRESETS.items()))
    ap.add_argument("--floor", choices=["none"] + list(FLOORS), default=None,
                    help="discard everything below the inferior border of this structure "
                         "(default: none, or the preset's floor)")
    ap.add_argument("--name", default=None, metavar="BASENAME",
                    help="output basename (default: soft_tissue_threshold, or the preset's)")
    ap.add_argument("--skin-mm", type=float, default=2.0, metavar="MM",
                    help="thickness of the skin layer, in mm inward from the body surface "
                         "(default 2.0; 0 disables the skin/fat/muscle breakdown)")
    ap.add_argument("--fat-hu-max", type=float, default=-30.0, metavar="HU",
                    help="upper HU bound for fat; above this counts as lean tissue "
                         "(default -30, the usual body-composition cutoff)")
    args = ap.parse_args()

    preset = PRESETS[args.preset] if args.preset else None

    # Explicit flags always win over the preset, which wins over the built-in defaults.
    if args.exclude is not None:
        enabled = list(args.exclude)
    elif preset:
        enabled = list(preset["exclude"])
    else:
        enabled = list(DEFAULT_EXCLUSIONS)
    enabled = [g for g in enabled if g not in args.no_exclude]

    floor = args.floor if args.floor is not None else (preset["floor"] if preset else "none")
    out_name = args.name or (preset["name"] if preset else "soft_tissue_threshold")
    label = "facial_soft_tissue" if preset and args.preset == "face" else "soft_tissue"

    seg_out = seg_dir_for(args.group) / args.case
    if not seg_out.is_dir():
        sys.exit(f"No segmentation output dir: {seg_out} (run segment_structures.py first)")

    ct_path = find_source_nifti(args.group, args.case)
    if not ct_path or not Path(ct_path).exists():
        sys.exit(f"No converted CT NIfTI for {args.group}/{args.case}")

    log(f"case      : {args.group} / {args.case}")
    log(f"seg dir   : {seg_out}")
    if preset:
        log(f"preset    : {args.preset} - {preset['desc']}")
    log(f"exclusions: {', '.join(enabled) if enabled else '(none)'}"
        + (f"  [+{args.exclude_dilate} vox global dilation]" if args.exclude_dilate else ""))
    log(f"floor     : {floor if floor != 'none' else '(none)'}")
    log(f"output    : {out_name}")

    log(f"loading CT: {os.path.basename(str(ct_path))}")
    ct_img = nib.load(str(ct_path))
    ref_shape = ct_img.shape
    zooms = ct_img.header.get_zooms()[:3]
    voxel_ml = float(np.prod(zooms)) / 1000.0
    log(f"shape={ref_shape}  spacing={tuple(round(z, 3) for z in zooms)} mm  "
        f"({voxel_ml*1000:.4f} mm^3/voxel)", 1)

    log("reading HU array into memory", 1)
    hu = ct_img.get_fdata(dtype=np.float32)

    # ---- envelope ----------------------------------------------------------
    log("STEP 1/5  body envelope")
    body_dir = ensure_task(args.group, seg_out, BODY_TASK, args.device)
    body = build_body(body_dir, ref_shape, verbose=True)

    air_in_body = body & (hu < args.hu_min)

    # Raw HU-window soft tissue, before any anatomic exclusion
    log(f"STEP 2/5  HU window {args.hu_min:.0f}..{args.hu_max:.0f} inside envelope")
    raw = body & (hu > args.hu_min) & (hu < args.hu_max)

    # ---- anatomic exclusions ------------------------------------------------
    # Segment the tasks the enabled groups need, before touching any mask.
    log("STEP 3/5  anatomic exclusions")
    needed = [t for g in enabled for t in EXCLUSION_GROUPS[g].get("requires", [])]
    if floor != "none":
        needed.append(FLOORS[floor]["source"][0])
    for task in dict.fromkeys(needed):
        log(f"prerequisite task: {task}", 1)
        ensure_task(args.group, seg_out, task, args.device)

    ml = lambda m: round(int(m.sum()) * voxel_ml, 1)

    exclude_mask = np.zeros(ref_shape, dtype=bool)
    group_report = {}
    for g, spec in EXCLUSION_GROUPS.items():
        on = g in enabled
        log(f"{'[ON ]' if on else '[off]'} {g}", 1)
        gmask, used, missing = build_group(seg_out, spec["sources"], ref_shape,
                                           verbose=True, indent=3)
        # Per-group dilation: TS masks are inferred at coarser resolution and resampled back,
        # so the label edge sits inside the true structure. For bone especially, the leftover
        # partial-volume shell (bone averaged with soft tissue) lands in the HU window and
        # would survive as a rim around every bone.
        grow = int(spec.get("dilate", 0))
        if grow > 0 and gmask.any():
            log(f"dilating {grow} vox", 3)
            gmask = ndimage.binary_dilation(gmask, iterations=grow)
        overlap = gmask & raw            # the part that actually costs volume
        log(f"-> {ml(gmask):.1f} mL mask, {ml(overlap):.1f} mL in the HU-window mask"
            + ("" if on else "  (not subtracted)"), 3)
        group_report[g] = {
            "enabled": g in enabled,
            "desc": spec["desc"],
            "dilate": grow,
            "mask_ml": ml(gmask),
            "removed_ml": ml(overlap),
            "masks_used": used,
            "structures_not_available": sorted(set(missing)),
        }
        if g in enabled:
            exclude_mask |= gmask

    if args.exclude_dilate > 0 and exclude_mask.any():
        log(f"global dilation of combined exclusion mask: {args.exclude_dilate} vox", 1)
        exclude_mask = ndimage.binary_dilation(exclude_mask, iterations=args.exclude_dilate)

    soft = raw & ~exclude_mask

    excluded_ml = round((int(raw.sum()) - int(soft.sum())) * voxel_ml, 1)

    floor_mask, floor_z, floor_cut_ml = None, None, 0.0
    if floor != "none":
        floor_mask, floor_z = build_floor(seg_out, floor, ct_img.affine, ref_shape)
        if floor_mask is not None:
            floor_cut_ml = ml(soft & ~floor_mask)
            soft = soft & floor_mask
            log(f"cut {floor_cut_ml:.1f} mL below the {FLOORS[floor]['desc']}", 1)

    # ---- tissue composition -------------------------------------------------
    log("STEP 4/5  skin / fat / muscle breakdown")
    if args.skin_mm > 0:
        log(f"skin = within {args.skin_mm:g} mm of the body surface and > {args.fat_hu_max:.0f} HU", 1)
    else:
        log("skin layer disabled (--skin-mm 0)", 1)
    skin, fat, muscle = build_composition(soft, body, hu, zooms,
                                          args.skin_mm, args.fat_hu_max)
    comp_ml = {"skin": ml(skin), "fat": ml(fat), "muscle": ml(muscle)}
    for k, v in comp_ml.items():
        log(f"{k:<7}{v:>9.1f} mL", 2)

    log("STEP 5/5  writing outputs")
    log(f"{out_name} (.nii.gz + .seg.nrrd)", 1)
    write_outputs(soft, out_name, seg_out, ct_img.affine, label=label)

    # One multilabel file so skin / fat / muscle can be shown and measured separately
    # in Slicer. The three are disjoint and sum exactly to the envelope above.
    comp = np.zeros(ref_shape, dtype=np.uint8)
    comp[muscle] = 3
    comp[fat] = 2
    comp[skin] = 1
    comp_img = nib.Nifti1Image(comp, ct_img.affine)
    nib.save(comp_img, str(seg_out / f"{out_name}_composition.nii.gz"))
    multilabel_to_segnrrd(comp_img, {1: "skin", 2: "fat", 3: "muscle"},
                          seg_out / f"{out_name}_composition.seg.nrrd")
    log(f"{out_name}_composition (.nii.gz + .seg.nrrd)", 1)

    stats = {
        "ct": os.path.basename(str(ct_path)),
        "voxel_ml": round(voxel_ml, 6),
        "body_envelope_ml": ml(body),
        "air_inside_body_ml": ml(air_in_body),
        "hu_window": [args.hu_min, args.hu_max],
        "preset": args.preset,
        "floor": None if floor == "none" else {
            "name": floor, "desc": FLOORS[floor]["desc"],
            "min_world_z_mm": None if floor_z is None else round(floor_z, 2),
            "applied": floor_mask is not None,
            "cut_below_floor_ml": floor_cut_ml},
        "exclusions_enabled": enabled,
        "exclude_dilate_voxels": args.exclude_dilate,
        "exclusion_groups": group_report,
        "composition": {"file": f"{out_name}_composition",
                        "skin_mm": args.skin_mm,
                        "fat_hu_max": args.fat_hu_max,
                        "skin_ml": comp_ml["skin"],
                        "fat_ml": comp_ml["fat"],
                        "muscle_ml": comp_ml["muscle"]},
        "soft_tissue": {"file": out_name,
                        "ml": ml(soft),
                        "ml_before_exclusions": ml(raw),
                        "excluded_ml": excluded_ml,
                        "floor_cut_ml": floor_cut_ml},
    }
    with open(seg_out / f"{out_name}.stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log(f"{out_name}.stats.json", 1)

    print("\n================ soft-tissue summary ================")
    print(f"  body envelope        : {stats['body_envelope_ml']:>8.1f} mL")
    print(f"  air inside body      : {stats['air_inside_body_ml']:>8.1f} mL")
    print(f"  HU window only       : {stats['soft_tissue']['ml_before_exclusions']:>8.1f} mL  "
          f"(HU {args.hu_min:.0f}..{args.hu_max:.0f}, in body)")
    print(f"  SOFT TISSUE          : {stats['soft_tissue']['ml']:>8.1f} mL  "
          f"(-{stats['soft_tissue']['excluded_ml']:.1f} mL exclusions"
          + (f", -{floor_cut_ml:.1f} mL below the {FLOORS[floor]['desc']}"
             if floor_mask is not None else "") + ")")
    print(f"    - skin              : {comp_ml['skin']:>8.1f} mL  "
          f"(<= {args.skin_mm:g} mm from the surface, > {args.fat_hu_max:.0f} HU)")
    print(f"    - fat               : {comp_ml['fat']:>8.1f} mL  "
          f"(<= {args.fat_hu_max:.0f} HU)")
    print(f"    - muscle / lean     : {comp_ml['muscle']:>8.1f} mL")
    print("\n  exclusion groups (volume overlapping the HU-window mask):")
    for g, r in group_report.items():
        flag = "ON " if r["enabled"] else "off"
        absent = r["structures_not_available"]
        if not r["masks_used"]:
            shown = ", ".join(absent[:3]) + (", ..." if len(absent) > 3 else "")
            print(f"    [{flag}] {g:<15} {'--':>8}      no masks available ({shown})")
        else:
            miss = (f"  [{len(absent)} structure(s) not produced - outside FOV?]"
                    if absent else "")
            dil = f" +{r['dilate']}vox" if r["dilate"] else ""
            print(f"    [{flag}] {g:<15} {r['removed_ml']:>8.1f} mL{dil}  "
                  f"({r['desc']}){miss}")
    print(f"\n  outputs -> {seg_out}")
    print("=====================================================")


if __name__ == "__main__":
    main()
