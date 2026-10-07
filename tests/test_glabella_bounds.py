"""Glabella and opisthocranion stay on the braincase.

A skull shell round a brain space, plus a "nose" below the brain's front that reaches
further forward than the brow, and a "headrest" labelled as skull well behind the head.
The length has to run brow to back of skull, ignoring both.

    python -m tests.test_glabella_bounds
"""
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from ctseg.segment_fossae import outer_measurements

n = 120                                   # 2 mm voxels, x = left-right, y = forward, z = up
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = -n                        # centre of the grid at the world origin
g = np.indices((n, n, n)).astype(float)
x, y, z = ((g[i] - n / 2) * 2.0 for i in range(3))
r = np.sqrt(x ** 2 + (y / 1.2) ** 2 + z ** 2)      # longer front to back than across

icv = r < 70
skull = (r >= 70) & (r < 76)                          # outer table: 76 mm, 91 mm forward
skull &= z > -30                                      # an open base, as a real skull is
skull |= (np.abs(x) < 6) & (y > 60) & (y < 125) & (z > -60) & (z < -40)   # the nose
skull |= (np.abs(x) < 40) & (y < -150) & (y > -170) & (z > -20) & (z < 20)  # headrest

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "skull.nii.gz"
    nib.save(nib.Nifti1Image(skull.astype(np.uint8), affine), str(p))
    up, lr, fwd = np.eye(3)[2], np.eye(3)[0], np.eye(3)[1]
    out = outer_measurements(p, None, None, None, None, affine, np.zeros(3),
                             up, lr, fwd, 0.0, icv=icv)

front, back = out["points"]["glabella"], out["points"]["opisthocranion"]
assert front[2] > -30, f"glabella went down to the nose: {front}"
assert back[1] > -100, f"opisthocranion went back to the headrest: {back}"
assert abs(out["length_ofd"] - 2 * 76 * 1.2) < 6, out["length_ofd"]

print("glabella bounds: ok")

# and the table carries the index with the width and length it is made of, no more
from ctseg.produce_table import read_stats
cols = dict(read_stats("CASE_cranial_linear", {"outer_mm": out}))
assert set(cols) == {("cranial", "index"), ("cranial", "width_mm"),
                     ("cranial", "length_mm")}, sorted(cols)
assert cols[("cranial", "length_mm")] == out["length_ofd"]
print("cranial index columns: ok")
