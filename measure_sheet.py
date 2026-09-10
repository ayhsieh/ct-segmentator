#!/usr/bin/env python
"""
Every linear measurement, drawn on the slice it was actually taken on, one page
per case, in a single PDF for reviewing a whole project by eye.

The point of the sheet is to catch a measurement whose number is fine but whose
line is not - a width that jumped to the other side of the head, a height that
ran past the fossa floor. So each panel is not a scanner slice with a line drawn
near it: the CT is re-cut along the measurement's own plane, in the head's own
frame, so the line lies flat in the picture and either touches bone at both ends
or visibly does not.

    python measure_sheet.py --project fossa
    python measure_sheet.py --project fossa demo --out sheet.pdf
    python measure_sheet.py --project fossa --case pvdo3a pvdo4a

Writes <project>_linear_measurements.pdf into the project's folder unless --out
says otherwise. Nothing is recomputed; the points are read as the pipeline left
them.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

APP = Path(__file__).resolve().parent
os.environ.setdefault("CT_DATA_ROOT", str(APP / "projects"))

from ct_paths import seg_dir_for                      # noqa: E402
from produce_table import stats_files                 # noqa: E402
from segment_structures import find_source_nifti      # noqa: E402
from segment_fossae import find_out, LABEL_VALUES     # noqa: E402
from ct_gui import MEASURES, RATIOS                   # noqa: E402

HALF = 105.0        # mm each way from the middle of the line
STEP = 0.6          # mm per pixel in the re-cut picture
WL, WW = 300.0, 1500.0                                # bone window
COLS, ROWS = 4, 4

# The compartment a regional measurement belongs to gets the colour that
# measurement is drawn in, so "does the line sit in its own fossa" is one look
# and not a cross-reference against a key.
ZONE_COLOUR = {LABEL_VALUES["anterior_fossa"]: "#c88bff",
               LABEL_VALUES["middle_fossa"]: "#ffb648",
               LABEL_VALUES["posterior_fossa"]: "#7de0d0"}
ZONE_NAME = {LABEL_VALUES["anterior_fossa"]: "anterior",
             LABEL_VALUES["middle_fossa"]: "middle",
             LABEL_VALUES["posterior_fossa"]: "posterior"}
# Light, because the bone under it is the thing being checked. The edge of each
# compartment is drawn as a line instead, which is where the question actually
# lives: a regional height is meant to stop on its own fossa floor.
WASH = 0.15


# What each kind of measurement runs along, what the other picture axis should be,
# and what such a cut is called. A width paired with forward gives a level cut, which
# is where you can see whether both ends sit on the widest part of the vault; a length
# and a height both want up in the picture, so a head still looks like a head.
KINDS = (("height", "up", "left_right", "coronal"),
         ("length", "forward", "up", "sagittal"),
         ("width", "left_right", "forward", "axial"))


def cut_axes(frame, key, ends):
    """Across and up for the cut, one of them lying exactly along the drawn line.

    Two fixed head axes would be tidier, but a line only lies in such a plane when
    it is exactly on axis, and several of these are not - maximum length is glabella
    to opisthocranion, which slopes down the back of the head. Off-plane by a
    centimetre and the picture shows the cut passing beside an end of the line
    rather than through it, which is the one thing this sheet exists to show.

    So the line claims an axis, and which axis it claims comes from what it is:
    a height claims the vertical one and stays vertical, a width and a length claim
    the horizontal one. The head then fills in the other, square to the line.
    """
    along, other, view = next((a, o, v) for k, a, o, v in KINDS if k in key)
    v = np.asarray(ends[-1], float) - np.asarray(ends[0], float)
    n = np.linalg.norm(v)
    line = v / n if n > 1e-6 else np.array(frame[along], float)
    if line @ np.array(frame[along], float) < 0:    # same way up as the anatomy
        line = -line
    rest = np.array(frame[other], float)
    rest = rest - (rest @ line) * line
    m = np.linalg.norm(rest)
    if m < 1e-6:                                    # degenerate; any square axis will do
        rest = np.cross(line, np.array(frame["up"], float))
        m = np.linalg.norm(rest) or 1.0
    rest = rest / m
    # a height is the vertical axis of its picture, everything else the horizontal one
    return (rest, line, view) if along == "up" else (line, rest, view)


def recut(vol, aff, centre, e1, e2, order=1, cval=-1024.0):
    """One oblique slice through centre, spanned by e1 and e2, in millimetres."""
    t = np.arange(-HALF, HALF + STEP, STEP)
    # rows are e2 and columns e1, which is the order imshow draws in: row to y,
    # column to x. Building it the other way round and transposing on the way to
    # the screen puts the picture at right angles to the line drawn over it, and
    # a head is round enough that this still looks like a head.
    v, u = np.meshgrid(t, t, indexing="ij")
    world = (centre[None, None, :] + u[..., None] * e1 + v[..., None] * e2)
    inv = np.linalg.inv(aff)
    vox = world @ inv[:3, :3].T + inv[:3, 3]
    return map_coordinates(vol, [vox[..., 0], vox[..., 1], vox[..., 2]],
                           order=order, mode="constant", cval=cval)


def load_zones(project, case, shape):
    """The three fossae as the pipeline labelled them, on the CT's own grid.

    Returned only when it is that grid. A label volume from an older run of a
    different series would line up with nothing, and a wash of colour sitting a
    centimetre off the anatomy it claims is worse than no wash at all.
    """
    p = find_out(seg_dir_for(project) / case, case, "fossae_simple", ".nii.gz")
    if not p.exists():
        return None
    lab = nib.load(str(p))
    if lab.shape != shape:
        return None
    return np.asarray(lab.dataobj).astype(np.uint8)


def load_stats(project, case):
    d = seg_dir_for(project) / case
    for f in stats_files(d):
        if not f.name.endswith("fossae_simple.stats.json"):
            continue
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def paint_zones(ax, lab, aff, centre, e1, e2):
    """The three fossae washed over the bone, each in its measurement's colour.

    Nearest neighbour, because a compartment label is a name and halfway between
    anterior and middle is not a place. The outline is the same cut drawn again
    at its edge: a height is supposed to stop on its own fossa floor, and a wash
    alone leaves you guessing at a boundary the eye has to find under grey.
    """
    if lab is None:
        return
    pic = recut(lab, aff, centre, e1, e2, order=0, cval=0)
    rgba = np.zeros(pic.shape + (4,), np.float32)
    for value, hexcol in ZONE_COLOUR.items():
        m = pic == value
        if m.any():
            rgba[m, :3] = mcolors.to_rgb(hexcol)
            rgba[m, 3] = WASH
    ax.imshow(rgba, origin="lower", extent=[-HALF, HALF, -HALF, HALF],
              interpolation="nearest")
    t = np.arange(-HALF, HALF + STEP, STEP)
    for value, hexcol in ZONE_COLOUR.items():
        m = (pic == value).astype(float)
        if m.any():
            ax.contour(t, t, m, levels=[0.5], colors=hexcol, linewidths=0.6,
                       alpha=0.95)


def panel(ax, vol, aff, mm, pts, frame, key, a, b, label, colour, look, lab):
    """One measurement on its own cut. Returns how far the number is from the line.

    That gap is the check worth printing. The drawn line is a real distance
    through the head; the number is what the pipeline wrote for it. If they
    disagree, one of the two is wrong and the page says so before you have to
    squint at the anatomy.
    """
    if b is None:                                   # the circumference ring
        ends = np.array(pts.get(a) or [], float)
        if not len(ends):
            ax.set_axis_off()
            return None
        e1 = np.array(frame["left_right"], float)
        e2 = np.array(frame["forward"], float)
        view, drawn = "axial", None
    elif a not in pts or b not in pts:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "%s\nnot measured" % label, ha="center", va="center",
                fontsize=7, color="#888888", transform=ax.transAxes)
        return None
    else:
        ends = np.array([pts[a], pts[b]], float)
        e1, e2, view = cut_axes(frame, key, ends)
        drawn = float(np.linalg.norm(ends[1] - ends[0]))

    # The cut passes through the line; the picture is centred on the head, so
    # every panel frames the same skull and the panels can be read against
    # each other rather than each one being its own private close-up.
    seed = ends.mean(axis=0)
    d = look - seed
    centre = seed + (d @ e1) * e1 + (d @ e2) * e2

    img = recut(vol, aff, centre, e1, e2)
    ax.imshow(img, cmap="gray", vmin=WL - WW / 2, vmax=WL + WW / 2,
              origin="lower", extent=[-HALF, HALF, -HALF, HALF],
              interpolation="bilinear")
    paint_zones(ax, lab, aff, centre, e1, e2)

    d = ends - centre
    x, y = d @ e1, d @ e2
    if b is None:
        ax.plot(np.r_[x, x[:1]], np.r_[y, y[:1]], "-", lw=1.1, color=colour)
    else:
        ax.plot(x, y, "-", lw=1.3, color=colour, solid_capstyle="butt")
        ax.plot(x, y, "o", ms=3.0, color=colour, mec="white", mew=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444444")
        s.set_linewidth(0.5)
    gap = None if drawn is None else abs(drawn - mm)
    warn = " !" if gap is not None and gap > 2.0 else ""
    ax.set_title("%s   %.1f mm%s" % (label, mm, warn), fontsize=7.5, pad=2.5,
                 color="#b00000" if warn else "black")
    ax.text(0.015, 0.015, view, transform=ax.transAxes, fontsize=6,
            color="#9ad4ff", va="bottom")
    return gap


def numbers_block(ax, mm, gaps, lab):
    ax.set_axis_off()
    top = 0.97
    if lab is not None:
        for n, value in enumerate(sorted(ZONE_COLOUR)):
            ax.text(0.02, top - 0.055 * n, "█  " + ZONE_NAME[value] + " fossa",
                    transform=ax.transAxes, fontsize=7, va="top",
                    color=ZONE_COLOUR[value], family="monospace")
        top -= 0.055 * len(ZONE_COLOUR) + 0.03
    rows = []
    for key, label, unit in RATIOS:
        if mm.get(key) is not None:
            v = mm[key] * 100 if key == "point_of_max_width" else mm[key]
            rows.append("%-14s%6.1f %s" % (label, v, unit))
    if mm.get("width_height_up_fraction") is not None:
        rows.append("%-14s%6.0f %%" % ("widest sits",
                                       mm["width_height_up_fraction"] * 100))
    bad = sorted(((v, k) for k, v in gaps.items() if v > 2.0), reverse=True)
    if bad:
        rows += ["", "number does not match its line:"]
        rows += ["  %s off by %.0f mm" % (k, v) for v, k in bad[:4]]
    if mm.get("warning"):
        rows += ["", "flagged: " + mm["warning"][:60]]
    ax.text(0.02, top, "\n".join(rows), transform=ax.transAxes, fontsize=7,
            family="monospace", va="top",
            color="#b00000" if (bad or mm.get("warning")) else "#222222")


def page(pdf, project, case, stats):
    o = (stats or {}).get("outer_mm") or {}
    pts, frame = o.get("points"), o.get("frame")
    ct = find_source_nifti(project, case)
    if not pts or not frame or not ct:
        return False
    img = nib.load(str(ct))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    lab = load_zones(project, case, vol.shape)

    # the middle of the head, so every panel frames the same skull
    look = np.array([v for k, v in pts.items() if k != "ofc_ring"], float).mean(axis=0)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(11.0, 11.6))
    fig.suptitle("%s / %s" % (project, case), fontsize=11, y=0.985)
    flat = axes.ravel()
    gaps, i = {}, 0
    for key, a, b, label, colour in MEASURES:
        if o.get(key) is None or i >= len(flat) - 1:
            continue
        g = panel(flat[i], vol, img.affine, o[key], pts, frame, key, a, b,
                  label, colour, look, lab)
        if g is not None:
            gaps[label] = g
        i += 1
    numbers_block(flat[i], o, gaps, lab)
    for ax in flat[i + 1:]:
        ax.set_axis_off()
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    pdf.savefig(fig, dpi=110)
    plt.close(fig)
    return True


def check_recut():
    """Does the cut land the world point (u, v) on the pixel drawn at (u, v)?

    A re-cut of a head looks like a head whichever way round it is built, so a
    picture at right angles to the line drawn over it passes the eye and fails
    the anatomy. Asked of a made-up volume the question has an exact answer and
    no thresholds: fill one with a straight ramp, which linear interpolation
    reproduces to the last decimal, and cut it at an angle that shares no axis
    with the grid. Every pixel must equal the ramp read at the world point the
    drawing code would put that pixel at.
    """
    shape = (40, 42, 44)
    # float64, so the only error left is the one being looked for: in float32 a
    # ramp reaching 440000 carries a hundredth of rounding, and the smallest
    # mistake worth catching - a single voxel along the first axis - is 1.0
    g = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float64)
    vol = 1.0 * g[0] + 100.0 * g[1] + 10000.0 * g[2]
    c, s = np.cos(0.4), np.sin(0.4)
    aff = np.array([[1.4 * c, -0.9 * s, 0.0, -11.0],
                    [1.4 * s, 0.9 * c, 0.0, 7.0],
                    [0.0, 0.0, 2.1, -30.0],
                    [0.0, 0.0, 0.0, 1.0]])
    inv = np.linalg.inv(aff)
    # the middle of the made-up volume, so the cut is inside it and the test has
    # something to test - centred anywhere else, every sample falls off the grid,
    # reads air, agrees with nothing, and passes
    centre = aff[:3, :3] @ (np.array(shape, float) / 2) + aff[:3, 3]
    e1 = np.array([0.6, -0.8, 0.0])
    e2 = np.array([0.48, 0.36, 0.8])

    half, step = 6.0, 1.5
    t = np.arange(-half, half + step, step)
    saved = globals()["HALF"], globals()["STEP"]
    globals()["HALF"], globals()["STEP"] = half, step
    try:
        pic = recut(vol, aff, centre, e1, e2)
    finally:
        globals()["HALF"], globals()["STEP"] = saved

    worst, seen = 0.0, 0
    for r, v in enumerate(t):
        for cix, u in enumerate(t):
            vx = inv[:3, :3] @ (centre + u * e1 + v * e2) + inv[:3, 3]
            if (vx < 0).any() or (vx > np.array(shape) - 1).any():
                continue            # off the grid, where the cut reads air
            seen += 1
            want = 1.0 * vx[0] + 100.0 * vx[1] + 10000.0 * vx[2]
            worst = max(worst, abs(float(pic[r, cix]) - want))
    print("cut lands within %.4f over %d of %d samples"
          % (worst, seen, len(t) ** 2))
    return seen == len(t) ** 2 and worst < 1e-3


def selftest(project, case):
    """The cut in the abstract, then the same cut through a real head."""
    if not check_recut():
        raise SystemExit("the picture is not square with the line drawn on it")

    stats = load_stats(project, case)
    o = ((stats or {}).get("outer_mm") or {})
    ct = find_source_nifti(project, case)
    if not o.get("points") or not ct:
        raise SystemExit("%s / %s has no measurements to check" % (project, case))
    pts, frame = o["points"], o["frame"]
    img = nib.load(str(ct))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    inv = np.linalg.inv(img.affine)
    look = np.array([v for k, v in pts.items() if k != "ofc_ring"],
                    float).mean(axis=0)

    ok = n = 0
    worst = (0.0, "")
    for key, a, b, label, colour in MEASURES:
        if b is None or o.get(key) is None or a not in pts or b not in pts:
            continue
        ends = np.array([pts[a], pts[b]], float)
        e1, e2, view = cut_axes(frame, key, ends)
        seed = ends.mean(axis=0)
        d = look - seed
        centre = seed + (d @ e1) * e1 + (d @ e2) * e2
        pic = recut(vol, img.affine, centre, e1, e2)
        d = ends - centre
        col = (d @ e1 + HALF) / STEP
        row = (d @ e2 + HALF) / STEP
        for c, r, w in zip(col, row, ends):
            v = inv[:3, :3] @ w + inv[:3, 3]
            want = float(map_coordinates(vol, v.reshape(3, 1), order=1,
                                         mode="constant", cval=-1024.0)[0])
            got = float(map_coordinates(pic, np.array([[r], [c]]), order=1,
                                        mode="constant", cval=-1024.0)[0])
            n += 1
            ok += abs(got - want) < 30
            if abs(got - want) > worst[0]:
                worst = (abs(got - want), label)
    # On a real head the two readings differ a little wherever an endpoint sits on
    # a bone edge: the panel resamples the CT and is then read again, and 1500 HU
    # per millimetre turns a fraction of a pixel into a hundred HU. A picture at
    # right angles to its line reads air where there is bone, which is thousands.
    print("panel matches the CT at %d of %d endpoints" % (ok, n))
    if worst[1]:
        print("worst gap %.0f HU at %s" % worst)
    if worst[0] > 400:
        raise SystemExit("the picture and the line it carries do not agree")
    print("measure sheet : draws where it samples")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", nargs="+", default=["fossa"])
    ap.add_argument("--case", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="check the drawing against the CT and stop")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.project[0], (args.case or ["SAMPLE1"])[0])

    jobs = []
    for p in args.project:
        seg = seg_dir_for(p)
        if not seg.is_dir():
            print("no results folder for %s" % p)
            continue
        for d in sorted(x for x in seg.iterdir() if x.is_dir()):
            if args.case and d.name not in args.case:
                continue
            s = load_stats(p, d.name)
            if s and (s.get("outer_mm") or {}).get("points"):
                jobs.append((p, d.name, s))

    if not jobs:
        raise SystemExit("no case has linear measurements yet")
    out = Path(args.out) if args.out else (
        seg_dir_for(args.project[0]).parent /
        ("%s_linear_measurements.pdf" % args.project[0]))
    out.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    with PdfPages(out) as pdf:
        for n, (p, case, s) in enumerate(jobs, 1):
            print("[%3d/%d] %s / %s" % (n, len(jobs), p, case), flush=True)
            try:
                done += bool(page(pdf, p, case, s))
            except Exception as e:
                print("        FAILED %s: %s" % (type(e).__name__, e), flush=True)
        pdf.infodict()["Title"] = "cranial linear measurements"
    print("\n%d page(s) -> %s" % (done, out))


if __name__ == "__main__":
    main()
