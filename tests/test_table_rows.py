"""Does the table hold every case, including the ones with no results?

The case list used to come from the results folder, so a case that was set aside or
never converted simply had no row - which makes a cohort table look complete when it
is not. The row for such a case is mostly empty, and that is the point: the name and
the note are the record of why it is not there.

    python -m tests.test_table_rows
"""
import json
import os
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    os.environ["CT_DATA_ROOT"] = str(root)
    from ctseg import ct_paths
    ct_paths.DATA_ROOT = root
    from ctseg import produce_table
    grp = root / "study"
    (grp / "total_segmentor_results_study" / "DONE").mkdir(parents=True)
    (grp / "total_segmentor_results_study" / "STRAY").mkdir(parents=True)
    (grp / "project.json").write_text(json.dumps({
        "name": "study",
        # order matters: the table should follow the project, not the filesystem
        "cases": [{"case": "DONE"}, {"case": "SETASIDE"}, {"case": "NEVERRAN"}],
        "skipped": ["SETASIDE"],
        "notes": {"SETASIDE": "no soft tissue scan", "NEVERRAN": "drive was offline"},
    }))

    assert produce_table.project_cases("study") == ["DONE", "SETASIDE", "NEVERRAN"]
    assert produce_table.case_notes("study")["SETASIDE"] == "no soft tissue scan"

    # a group with no project.json falls back to whatever results exist
    (root / "cli" / "total_segmentor_results_cli" / "ONLY").mkdir(parents=True)
    assert produce_table.project_cases("cli") == []
    assert produce_table.case_notes("cli") == {}

print("table rows: ok")
