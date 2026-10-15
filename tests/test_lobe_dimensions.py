"""Lobe dimensions: the midline is fitted through the midline structures, so a lobe that
is bigger on one side measures bigger on that side, and the head's tilt changes nothing.

    python -m tests.test_lobe_dimensions
"""
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from ctseg import lobe_dimensions as ld
from ctseg.produce_table import read_stats

n = 64                                     # 2 mm voxels; x = right, y = front, z = up
g = np.indices((n, n, n))
x, y, z = ((g[i] - n / 2) * 2.0 for i in range(3))


def head(tilt_deg):
    """Ellipsoid cerebrum split into lobes, right half 20% wider than the left."""
    t = np.radians(tilt_deg)                       # roll the head about the y axis
    xr, zr = x * np.cos(t) - z * np.sin(t), x * np.sin(t) + z * np.cos(t)
    xs = np.where(xr > 0, xr / 1.2, xr)            # the right side is wider
    brain = (xs / 45) ** 2 + (y / 55) ** 2 + (zr / 40) ** 2 < 1
    m = {"frontal_lobe": brain & (y > 15), "parietal_lobe": brain & (y <= 15) & (y > -25) & (zr > 0),
         "temporal_lobe": brain & (y <= 15) & (y > -25) & (zr <= 0),
         "occipital_lobe": brain & (y <= -25),
         "septum_pellucidum": (np.abs(xr) < 1.5) & (np.abs(y - 10) < 12) & (np.abs(zr) < 10),
         "brainstem": (xr ** 2 + (y + 5) ** 2 < 36) & (zr < -10) & (zr > -40),
         "venous_sinuses": (np.abs(xr) < 2) & (zr > 30) & brain}
    return m


aff = np.diag([2.0, 2.0, 2.0, 1.0])
aff[:3, 3] = -n
dims = []
for tilt in (0, 15):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for k, v in head(tilt).items():
            nib.save(nib.Nifti1Image(v.astype(np.uint8), aff), str(d / f"{k}.nii.gz"))
        (c, nrm, rms), why = ld.midline(d)
        assert why is None and abs(abs(nrm[0]) - np.cos(np.radians(tilt))) < 0.05, nrm
        fwd, up = ld.frame(d, c, nrm)
        par = ld.world(np.asarray(nib.load(str(d / "parietal_lobe.nii.gz")).dataobj) > 0, aff)
        side = (par - c) @ nrm
        w = [np.ptp(np.abs((par[side > 0] - c) @ nrm)), np.ptp(np.abs((par[side < 0] - c) @ nrm))]
        dims.append(w)
        assert w[0] > 1.1 * w[1], w                # right measures wider, as built
assert abs(dims[0][0] - dims[1][0]) < 4, dims       # and tilting the head changes little

cols = dict(read_stats("lobe_dimensions", {"lobes": {"frontal_lobe": {
    "partial": False, "right": {"width_mm": 50.0}, "left": {"width_mm": 48.0}}}}))
assert cols == {("lobe", "frontal_right_width_mm"): 50.0, ("lobe", "frontal_left_width_mm"): 48.0,
                ("lobe", "frontal_partial"): 0}, cols
print("lobe dimensions: ok")
