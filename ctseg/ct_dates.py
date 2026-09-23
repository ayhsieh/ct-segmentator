"""The study date of the series a case was segmented from.

A case folder can hold more than one study, so the date is taken from the series that
was actually converted and segmented - the one recorded in the series selection cache -
rather than from whatever DICOM turns up first. Without a recorded choice it falls back
to reading the folder, and says so.

Headers only, never pixel data, and pydicom rather than segment_structures so a CSV
build does not load torch.
"""
import os
from collections import Counter
from pathlib import Path

from ctseg.ct_paths import cache_get, case_dir_for, load_cache, unanchored

DATE_TAGS = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"]


def iso(d):
    """DICOM dates are YYYYMMDD; anything else is passed through untouched."""
    d = str(d or "").strip()
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d


def _date_of(ds):
    for tag in DATE_TAGS:
        v = getattr(ds, tag, "")
        if v:
            return iso(v)
    return ""


def _recorded_series(group, case_dir):
    """The folder of the series that was chosen for this case, if one was.

    Through ct_paths, which knows where a group keeps its choices and how a case's
    path is recorded there.
    """
    entry = cache_get(load_cache(group), case_dir, group)
    if not isinstance(entry, dict) or not entry.get("series_dir"):
        return None
    d = unanchored(entry["series_dir"], group)
    return d if d.is_dir() else None


def study_date(group, case, limit=400, case_dir=None):
    """(date, note). The note is empty when the date came from the chosen series.

    case_dir is for callers that already know where the case is; everyone else gets
    the folder worked out from the group and the case name.
    """
    import pydicom
    case_dir = Path(case_dir) if case_dir else case_dir_for(group, case)
    if not case_dir.exists():
        return "", "case folder not found"

    series = _recorded_series(group, case_dir)
    if series is not None:
        for f in sorted(series.iterdir()):
            if not f.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True,
                                     specific_tags=DATE_TAGS)
            except Exception:
                continue
            d = _date_of(ds)
            if d:
                return d, ""            # one slice is enough; a series has one date
        return "", "no date in the chosen series"

    # No recorded choice: read the folder and report what is there. More than one date
    # in a case folder usually means two timepoints were exported together, which is
    # worth seeing rather than silently picking one.
    seen, n = Counter(), 0
    for root, _, files in os.walk(case_dir):
        for fname in files:
            if n >= limit:
                break
            try:
                ds = pydicom.dcmread(os.path.join(root, fname),
                                     stop_before_pixels=True, specific_tags=DATE_TAGS)
            except Exception:
                continue
            n += 1
            d = _date_of(ds)
            if d:
                seen[d] += 1
    if not seen:
        return "", "no DICOM dates found"
    date, _ = seen.most_common(1)[0]
    if len(seen) > 1:
        return date, f"no series chosen; folder holds {len(seen)} dates: " + \
                     ", ".join(sorted(seen))
    return date, "no series chosen; read from the folder"
