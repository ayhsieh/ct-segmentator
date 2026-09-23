"""Does a case note go on and come off the project cleanly?

The interesting part is not storing a string, it is clearing one: an emptied note has
to be removed, or a project collects a key for every case anyone ever clicked on and
`notes` slowly fills with empty strings that read as "there is a note here".

    python -m tests.test_case_note
"""
from ctseg.ct_gui import set_note

pr = {"name": "demo", "cases": [{"case": "A"}, {"case": "B"}]}

assert set_note(pr, "A", "excluded: motion artefact through the vault") == \
    "excluded: motion artefact through the vault"
assert pr["notes"] == {"A": "excluded: motion artefact through the vault"}

set_note(pr, "B", "  second timepoint, same child  ")
assert pr["notes"]["B"] == "second timepoint, same child"      # trimmed

set_note(pr, "A", "")                                          # cleared, not blanked
assert "A" not in pr["notes"], pr["notes"]
assert pr["notes"] == {"B": "second timepoint, same child"}

set_note(pr, "A", "   ")                                       # whitespace is empty
assert "A" not in pr["notes"], pr["notes"]

set_note(pr, "B", None)                                        # nothing sent at all
assert pr["notes"] == {}

assert len(set_note(pr, "A", "x" * 5000)) == 2000              # capped, not rejected

print("case note: ok")
