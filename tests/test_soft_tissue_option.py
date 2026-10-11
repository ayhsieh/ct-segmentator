"""Soft tissue as a Run-tab option: it segments what it subtracts, and its numbers
reach the table.

    python -m tests.test_soft_tissue_option
"""
from ctseg.ct_gui import ANALYSES, prereq
from ctseg.produce_table import read_stats

from ctseg.segment_soft_tissue import EXCLUSION_GROUPS, FLOORS, PRESETS

opt = ANALYSES["soft_tissue"]
assert opt["args"] == ["--preset", "head"]
# every task the preset subtracts or floors on is a prerequisite step, not a surprise
# TotalSegmentator run inside the analysis
head = PRESETS["head"]
used = {t for g in head["exclude"] for t, _ in EXCLUSION_GROUPS[g]["sources"]}
used |= {"body", FLOORS[head["floor"]]["source"][0]}
assert used <= set(opt["needs"]), used - set(opt["needs"])
assert opt["proofs"] == [head["name"] + ".stats.json"]
assert prereq("head_muscles") == ([], ["head_muscles/*.nii.gz"])
assert prereq("total")[0] == ["--roi-subset", "brain", "skull"]

stats = {"soft_tissue": {"file": "head_soft_tissue", "ml": 593.9},
         "composition": {"skin_ml": 94.2, "fat_ml": 124.2, "muscle_ml": 375.5}}
cols = {f"{t}_{n}": v for (t, n), v in read_stats("head_soft_tissue", stats)}
assert cols == {"head_soft_tissue_ml": 593.9, "head_soft_tissue_skin_ml": 94.2,
                "head_soft_tissue_fat_ml": 124.2,
                "head_soft_tissue_muscle_ml": 375.5}, cols
print("soft tissue option: ok")
