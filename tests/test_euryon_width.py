"""Width across the inner cortex, levelled on Frankfort horizontal.

Two things are easy to get wrong and neither shows up in the number: the plane can
come out upside down, so "level" tilts the wrong way; and the width can be read off
the outer table, which is a different measurement that already has a column. The
check builds a box of known width, tilts the head, and asks for the width back.

    python -m tests.test_euryon_width
"""
import numpy as np

from ctseg.segment_fossae import frankfort_frame, width_euryon

up = np.array([0.0, 0.0, 1.0])

# porions level, orbitale in front and a little below: an upright Frankfort plane
hand = {"PR(L)": [-60.0, -20.0, 0.0], "PR(R)": [60.0, -20.0, 0.0],
        "OR(L)": [-25.0, 55.0, -2.0], "OR(R)": [25.0, 55.0, -2.0]}
across, n, fwd = frankfort_frame(hand, up)
assert n @ up > 0, "the plane normal has to point at the vault, not the jaw"
assert abs(across @ n) < 1e-6 and abs(fwd @ n) < 1e-6      # a genuine frame

assert frankfort_frame({"PR(L)": [0, 0, 0], "PR(R)": [1, 0, 0]}, up) is None
assert frankfort_frame({"OR(L)": [0, 0, 0]}, up) is None

# a block 40 voxels across at 2 mm a voxel: 78 mm between the outermost centres
icv = np.zeros((40, 30, 24), bool)
icv[:, 5:25, 4:20] = True
affine = np.diag([2.0, 2.0, 2.0, 1.0])
affine[:3, 3] = [-40.0, -30.0, -24.0]

w, a, b, why = width_euryon(icv, affine, hand, up)
assert why == "" and w is not None, why
assert abs(w - 78.0) < 2.0, w
assert abs((a - b) @ across) > 70.0          # the pair really does span the head

# no hand-placed points: declined, and the reason says which ones
w2, _, _, why2 = width_euryon(icv, affine, {}, up)
assert w2 is None and "porion" in why2, why2

print("euryon width: ok")
