"""A case that was set aside keeps its row, but contributes no numbers.

Setting a case aside does not delete what was already computed for it, so the stats
files stay on disk. If the table reads them anyway, an excluded case goes on being
counted - which is how a duplicate import ends up in a cohort twice.

    python -m tests.test_set_aside
"""
import json
import os
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    os.environ["CT_DATA_ROOT"] = str(root)
    import ct_paths
    ct_paths.DATA_ROOT = root
    import produce_table

    grp = root / "study"
    res = grp / "total_segmentor_results_study"
    # both cases carry a result; only one of them is still in the cohort
    for case in ("KEEP", "DUPE"):
        (res / case).mkdir(parents=True)
        (res / case / "brain_icv.stats.json").write_text(
            json.dumps({"case": case, "status": "ok", "brain_ml": 1.0, "icv_ml": 2.0}))
    (grp / "project.json").write_text(json.dumps({
        "name": "study",
        "cases": [{"case": "KEEP"}, {"case": "DUPE"}],
        "skipped": ["DUPE"],
        "notes": {"DUPE": "duplicate import of KEEP"},
    }))

    assert produce_table.set_aside("study") == {"DUPE"}
    # the row survives, so the record of the exclusion survives with it
    assert produce_table.project_cases("study") == ["KEEP", "DUPE"]
    assert produce_table.case_notes("study")["DUPE"] == "duplicate import of KEEP"
    # a group with no project.json sets nothing aside
    assert produce_table.set_aside("nosuch") == set()

print("set aside: ok")
