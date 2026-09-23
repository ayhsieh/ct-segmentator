"""Does a Slicer markups file become a landmark CSV the loader will read?

Two things silently lose a whole case: the coordinate system flipping (Slicer writes
either LPS or RAS, the loader assumes LPS), and a label the loader does not know, which
it drops without a word. Both are checked here by round-tripping through the real
loader rather than by reading the file back.

    python -m tests.test_markups_points
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    os.environ["CT_DATA_ROOT"] = str(root)
    import ct_paths
    ct_paths.DATA_ROOT = root
    import markups_to_points as m2p
    from segment_fossae import load_manual_landmarks

    def markups(system, points):
        return {"markups": [{"coordinateSystem": system, "controlPoints": [
            {"label": k, "position": list(v)} for k, v in points.items()]}]}

    pts = {"n": [10.0, -90.0, 20.0], "s": [1.0, 5.0, -10.0],
           "PR(L)": [-60.0, -20.0, 0.0], "PR(R)": [60.0, -20.0, 0.0],
           "OR(L)": [-25.0, 55.0, -2.0], "OR(R)": [25.0, 55.0, -2.0],
           "UR3OI": [0.0, 0.0, 0.0]}          # a label from some other analysis
    src = root / "SAMPLE1.mrk.json"
    src.write_text(json.dumps(markups("LPS", pts)))

    (root / "study").mkdir()
    assert m2p.main.__module__                 # imported, not run
    import sys
    sys.argv = ["x", "--group", "study", str(src)]
    assert m2p.main() == 0

    got = load_manual_landmarks("study", "SAMPLE1")
    assert set(got) == set(pts) - {"UR3OI"}, sorted(got)   # the stray label is dropped
    # the loader hands back RAS, so LPS in means the first two axes flip
    assert np.allclose(got["n"], [-10.0, 90.0, 20.0]), got["n"]
    assert np.allclose(got["PR(R)"], [-60.0, 20.0, 0.0]), got["PR(R)"]

    # a file Slicer wrote in RAS has to come out at the same place in the end
    src2 = root / "TCIA1.mrk.json"
    ras = {k: [-v[0], -v[1], v[2]] for k, v in pts.items()}
    src2.write_text(json.dumps(markups("RAS", ras)))
    sys.argv = ["x", "--group", "study", str(src2)]
    assert m2p.main() == 0
    got2 = load_manual_landmarks("study", "TCIA1")
    assert np.allclose(got2["n"], got["n"]), (got2["n"], got["n"])

    # --case files a differently-named export under the case it belongs to
    src3 = root / "whatever_export.mrk.json"
    src3.write_text(json.dumps(markups("LPS", pts)))
    sys.argv = ["x", "--group", "study", "--case", "AB", str(src3)]
    assert m2p.main() == 0
    assert set(load_manual_landmarks("study", "AB")) == set(pts) - {"UR3OI"}

print("markups to points: ok")
