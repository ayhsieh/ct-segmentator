"""Facial soft tissue as a Run-tab option: it segments what it subtracts, and its
numbers reach the table.

    python -m tests.test_soft_tissue_option
"""
from ctseg.ct_gui import ANALYSES, prereq
from ctseg.produce_table import read_stats

face = ANALYSES["facial_soft_tissue"]
assert face["args"] == ["--preset", "face"]
# every task the face preset subtracts is a prerequisite step, not a surprise
# TotalSegmentator run inside the analysis
for t in ("body", "brain_structures", "head_glands_cavities", "head_muscles",
          "craniofacial_structures"):
    assert t in face["needs"], t
assert prereq("head_muscles") == ([], ["head_muscles/*.nii.gz"])
assert prereq("total")[0] == ["--roi-subset", "brain", "skull"]

stats = {"soft_tissue": {"file": "facial_soft_tissue", "ml": 593.9},
         "composition": {"skin_ml": 94.2, "fat_ml": 124.2, "muscle_ml": 375.5}}
cols = {f"{t}_{n}": v for (t, n), v in read_stats("facial_soft_tissue", stats)}
assert cols == {"facial_soft_tissue_ml": 593.9, "facial_soft_tissue_skin_ml": 94.2,
                "facial_soft_tissue_fat_ml": 124.2,
                "facial_soft_tissue_muscle_ml": 375.5}, cols
print("soft tissue option: ok")
