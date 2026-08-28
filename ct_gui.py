#!/usr/bin/env python
"""
A local web front-end for the segmentation pipeline.

Wraps segment_structures.py, segment_fossae.py, brain_icv.py and produce_table.py so
they can be driven from a browser: pick a folder of DICOMs, choose what to segment,
watch it run, download a CSV. Nothing leaves the machine - the server binds the
loopback interface only and every request carries a token minted at startup.

    python ct_gui.py                 # then open the URL it prints
    python ct_gui.py --open          # and open the browser for you

Your DICOMs are never copied. A project registers the folder in place, and everything
generated - converted NIfTI, segmentations, CSVs - lands under projects/ next to this
file.

Two extra modes exist for the server's own use, not for you:
    python ct_gui.py --scan PATH     # score the DICOM series in PATH, print JSON
    python ct_gui.py --selftest      # report interpreter, packages, GPU
"""
import argparse
import ast
import json
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

APP = Path(__file__).resolve().parent
PROJECTS = APP / "projects"
PAGE_FILE = APP / "ct_gui_page.html"
TOKEN = secrets.token_urlsafe(18)
WIN = sys.platform == "win32"

# Folders the pipeline creates inside a group - never offer these as cases
RESERVED = ("converted_nifti_", "total_segmentor_results_", "points", "logs")
MAX_LOG_LINES = 4000


# ------------------------------------------------------------------ the pipeline
# Read from the pipeline source rather than hardcoded, so the two cannot drift, and
# by parsing rather than importing, because importing segment_structures pulls in
# torch (~2s and a lot of memory) and we want the server to start instantly and to
# never hold a CUDA context - every real job runs in its own subprocess.
def _literal(path, *names):
    out = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


_T = _literal(APP / "segment_structures.py", "AVAILABLE_TASKS", "LICENSED_TASKS")
AVAILABLE_TASKS = _T.get("AVAILABLE_TASKS", [])
LICENSED_TASKS = _T.get("LICENSED_TASKS", [])
FBIN_MM = _literal(APP / "segment_fossae.py", "FBIN_MM").get("FBIN_MM", 1.5)

# One-line descriptions so the list means something to someone who has not read the
# TotalSegmentator paper. Anything unlisted still appears, just without a blurb.
BLURB = {
    "total": "117 structures across the whole body - organs, bones, muscles, vessels",
    "total_mr": "the whole-body set, for MR instead of CT",
    "body": "body outline, trunk, extremities and skin",
    "body_mr": "body outline and extremities, for MR",
    "vertebrae_mr": "individual vertebrae C1-L5 and the sacrum, for MR",
    "lung_vessels": "pulmonary arteries, veins, airways and airway walls",
    "lung_nodules": "lung nodules",
    "cerebral_bleed": "intracerebral haemorrhage",
    "brain_aneurysm": "intracranial aneurysms - TOF MR only, not CT",
    "ventricle_parts": "the ventricles split into their parts",
    "hip_implant": "hip prostheses",
    "pleural_pericard_effusion": "pleural and pericardial effusion",
    "liver_vessels": "hepatic vessels and tumours",
    "liver_segments": "the Couinaud liver segments",
    "liver_segments_mr": "the Couinaud liver segments, for MR",
    "kidney_cysts": "renal cysts",
    "breasts": "breast tissue",
    "head_glands_cavities": "eyes, glands and the air cavities of the head",
    "head_muscles": "facial and masticatory muscles",
    "headneck_bones_vessels": "laryngeal structures, cartilage and neck vessels",
    "headneck_muscles": "neck and pharyngeal muscles",
    "oculomotor_muscles": "the extraocular muscles and optic nerve",
    "craniofacial_structures": "mandible, teeth, skull and sinuses",
    "teeth": "the teeth individually",
    "abdominal_muscles": "the core and torso muscles",
    "trunk_cavities": "the abdominal and thoracic cavities, and the mediastinum",
    "brain_structures": "brain regions: lobes, cerebellum, brainstem, ventricles, CSF",
    "face": "the face, for defacing/anonymising",
    "face_mr": "the face, for MR",
    "vertebrae_body": "vertebral bodies without the posterior elements",
    "heartchambers_highres": "the four heart chambers at high resolution",
    "coronary_arteries": "the coronary tree",
    "aortic_sinuses": "the aortic valve cusps and outflow tract",
    "appendicular_bones": "arm and leg bones",
    "appendicular_bones_mr": "arm and leg bones, for MR",
    "thigh_shoulder_muscles": "thigh and shoulder muscle groups",
    "thigh_shoulder_muscles_mr": "thigh and shoulder muscles, for MR",
    "tissue_types": "subcutaneous fat, torso fat, skeletal muscle",
    "tissue_types_mr": "the same tissue types, for MR",
    "tissue_4_types": "the tissue types plus intermuscular fat",
}

# Whether TotalSegmentator already holds a valid licence. Asked once, in a subprocess,
# so the answer costs nothing at startup and torch never loads into the server. When a
# licence is already stored the UI stops demanding a number for the licensed tasks -
# they will simply run, exactly as they do from the command line.
_LICENSE_STATE = None


def license_stored():
    global _LICENSE_STATE
    if _LICENSE_STATE is None:
        _LICENSE_STATE = False
        try:
            r = subprocess.run(
                [sys.executable, "-c", "from totalsegmentator.config import "
                 "has_valid_license_offline as h; print(h()[0])"],
                capture_output=True, text=True, timeout=60)
            _LICENSE_STATE = r.stdout.strip().endswith("yes")
        except Exception:
            pass
    return _LICENSE_STATE

# A browser cannot hand a page a folder path - the file input gives file contents, not
# a location, which is useless when the point is to segment a folder in place. But the
# server is on the same machine as the person clicking, so it can open the real OS
# folder chooser itself. It runs in a subprocess: Tk must own the main thread, and this
# one is a request thread, and a dialog that hangs then cannot take the server with it.
_PICKER = r"""
import sys, tkinter, tkinter.filedialog
r = tkinter.Tk(); r.withdraw()
r.attributes("-topmost", True)      # otherwise it opens behind the browser window
p = tkinter.filedialog.askdirectory(title="Choose the folder that holds your scans",
                                    initialdir=(sys.argv[1] or None), mustexist=True)
r.destroy()
sys.stdout.write(p or "")
"""


def native_folder_dialog(start=""):
    """Return (path, unavailable_reason). An empty path with no reason means cancelled."""
    try:
        r = subprocess.run([sys.executable, "-c", _PICKER, str(start)],
                           capture_output=True, text=True, timeout=600)
    except Exception as e:
        return "", str(e)
    if r.returncode != 0:
        # no display, or a python built without Tk - the built-in browser still works
        tail = (r.stderr or "").strip().splitlines()
        return "", (tail[-1] if tail else "no folder dialog available here")
    p = r.stdout.strip()
    return (p if p and Path(p).is_dir() else ""), ""


# The two derived analyses in this repo, which are not TotalSegmentator tasks
ANALYSES = {
    "fossae": {
        "label": "Cranial fossa volumes",
        "blurb": "anterior / middle / posterior compartments from the skull-floor map",
        "script": "segment_fossae.py",
        "needs": ["brain_structures"],
    },
    "brain_icv": {
        "label": "Brain and intracranial volume",
        "blurb": "parenchyma and ICV, as two Slicer layers plus a CSV",
        "script": "brain_icv.py",
        "needs": ["brain_structures"],
    },
}


def seg_dir_for(group):
    """Mirrors segment_structures.seg_dir_for; duplicated to avoid importing torch."""
    return PROJECTS / group / f"total_segmentor_results_{group}"


def nifti_dir_for(group):
    return PROJECTS / group / f"converted_nifti_{group}"


# ------------------------------------------------------------------- dicom probing
def is_dicom(path):
    """The same preamble check segment_structures.is_dicom_file uses."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


# DICOMDIR is a media index in DICOM format with no pixel data. It passes the preamble
# check, so a folder holding only an index and some subfolders looks like a series
# folder - which is how a study ends up registered one level too high.
INDEX_FILES = {"dicomdir"}


def _image_file(f):
    return (f.is_file() and not f.name.startswith(".")
            and f.name.lower() not in INDEX_FILES and is_dicom(str(f)))


def has_dicom_direct(d, limit=200):
    try:
        for i, f in enumerate(d.iterdir()):
            if i > limit:
                break
            if _image_file(f):
                return True
    except (PermissionError, OSError):
        pass
    return False


# Folders that never hold the images but can be enormous - a study exported to disc
# often ships its own viewer, and descending into it exhausts the search budget before
# the DICOMs are ever reached.
SKIP_DIRS = {"viewer.app", "reports", "report", "__macosx", "$recycle.bin",
             "system volume information"}


def has_dicom_anywhere(d, budget=6000):
    """Any DICOM image below d. Breadth-first, so shallow folders are checked before
    deep ones, with a budget so a huge tree cannot stall the browser."""
    from collections import deque as _dq
    q, seen = _dq([d]), 0
    while q and seen < budget:
        cur = q.popleft()
        try:
            entries = list(cur.iterdir())
        except (PermissionError, OSError):
            continue
        for f in entries:
            seen += 1
            if _image_file(f):
                return True
            if (f.is_dir() and not f.name.startswith(".")
                    and f.name.lower() not in SKIP_DIRS):
                q.append(f)
    return False


def series_root(case_dir):
    """The folder to register for a case: the one holding the series.

    This is the single most important function here. segment_structures.resolve_folders
    classifies whatever path it is given, and for a case laid out as
    CASE/dicom/SER0001/*.dcm it takes the "batch" branch and invents one case per
    subfolder - producing results directories literally named `dicom`, `reports` and
    `viewer.app`. Registering the series root instead means resolve_folders always sees
    a single study, and the junction's own name supplies the case name.

    Returns (path, note). note is non-empty when the layout was ambiguous.
    """
    p = Path(case_dir)
    if has_dicom_direct(p):
        return p, ""
    try:
        subs = [d for d in sorted(p.iterdir()) if d.is_dir()
                and not d.name.startswith(".")]
    except (PermissionError, OSError):
        return p, "could not read the folder"
    if not subs:
        return p, ""
    with_dicom = [d for d in subs if has_dicom_direct(d)]
    with_subs = [d for d in subs if any(x.is_dir() for x in d.iterdir())]
    if with_dicom and not with_subs:
        return p, ""                       # series folders sit directly under p
    leads = [d for d in subs if has_dicom_anywhere(d)]
    if len(leads) == 1:
        return series_root(leads[0])
    if not leads:
        return p, "no DICOM files found"
    return p, f"DICOMs under {len(leads)} subfolders - check this is one study"


def detect_cases(root):
    """Case folders under a candidate project folder."""
    root = Path(root)
    out = []
    try:
        subs = sorted(d for d in root.iterdir() if d.is_dir())
    except (PermissionError, OSError) as e:
        return out, f"{type(e).__name__}: {e}"
    for d in subs:
        if d.name.startswith(".") or d.name.startswith(RESERVED) or d.name in RESERVED:
            continue
        if not has_dicom_anywhere(d):
            continue
        sr, note = series_root(d)
        out.append({"case": d.name, "path": str(d), "series_root": str(sr),
                    "note": note})
    if not out and has_dicom_anywhere(root):
        sr, note = series_root(root)
        out.append({"case": root.name, "path": str(root), "series_root": str(sr),
                    "note": note})
    return out, ""


# ------------------------------------------------------------------------- links
def make_link(link, target):
    """A junction (Windows) or symlink (POSIX) so the pipeline sees the DICOMs in
    place. Junctions need no elevation, unlike `mklink /D` or os.symlink on Windows."""
    link, target = Path(link), Path(target)
    if link.exists() or link.is_symlink():
        return True, "already there"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if WIN:
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return False, (r.stderr or r.stdout).strip()
        else:
            os.symlink(str(target), str(link), target_is_directory=True)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def drop_link(link):
    """Remove a link without following it. rmtree through a junction would delete the
    user's DICOMs, so links are always taken out first and by name."""
    link = Path(link)
    try:
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            os.rmdir(link)              # a junction is an empty-looking dir to rmdir
    except OSError:
        pass


# ------------------------------------------------------------------- the registry
def project_file(name):
    return PROJECTS / name / "project.json"


def load_project(name):
    p = project_file(name)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    d["name"] = name
    return d


def save_project(d):
    p = project_file(d["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))


def list_projects():
    if not PROJECTS.is_dir():
        return []
    out = []
    for d in sorted(PROJECTS.iterdir()):
        if not d.is_dir():
            continue
        pr = load_project(d.name)
        if pr:
            out.append(pr)
    return out


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,60}$")


def case_status(group, case):
    """What exists on disk for this case. Never trust a job's exit code - several
    pipeline paths log a warning, skip the case and still exit 0 - so the truth is
    always re-derived from the filesystem."""
    d = seg_dir_for(group) / case
    done = []
    if d.is_dir():
        for f in d.glob("*.seg.nrrd"):
            stem = f.name[: -len(".seg.nrrd")]
            if stem in AVAILABLE_TASKS or stem in LICENSED_TASKS:
                done.append(stem)
    fossa = bool(list(d.glob("*fossae_simple.stats.json"))) if d.is_dir() else False
    icv = (d / "brain_icv.stats.json").exists() if d.is_dir() else False
    traced = bool(list(d.glob("*_traced.json"))) if d.is_dir() else False
    nii = nifti_dir_for(group) / case
    return {"case": case, "tasks": sorted(done), "fossae": fossa, "brain_icv": icv,
            "traced": traced,
            "converted": bool(list(nii.glob("*.nii.gz"))) if nii.is_dir() else False}


def project_state(name):
    pr = load_project(name)
    if not pr:
        return None
    cases = []
    for c in pr.get("cases", []):
        st = case_status(name, c["case"])
        link = PROJECTS / name / c["case"]
        st["online"] = link.exists()
        st["source"] = c.get("series_root", c.get("path", ""))
        cases.append(st)
    pr = dict(pr)
    pr["case_status"] = cases
    pr["seg_dir"] = str(seg_dir_for(name))
    return pr


# ---------------------------------------------------------------------- the jobs
class Job:
    def __init__(self, kind, project, label, steps, queue_name="gpu"):
        self.id = secrets.token_hex(6)
        self.kind = kind
        self.project = project
        self.label = label
        self.steps = steps              # list of (caption, argv) or ("fn", callable)
        self.queue = queue_name
        self.state = "queued"
        self.lines = deque(maxlen=MAX_LOG_LINES)
        self.seq = 0
        self.rc = None
        self.step_i = 0
        self.step_n = len(steps)
        self.step_label = ""
        self.created = time.time()
        self.started = None
        self.ended = None
        self.proc = None
        self.cancelled = False
        self.error = ""

    def emit(self, text):
        for line in str(text).rstrip("\n").split("\n"):
            self.seq += 1
            self.lines.append((self.seq, line))

    def snapshot(self, since=0):
        return {
            "id": self.id, "kind": self.kind, "project": self.project,
            "label": self.label, "state": self.state, "rc": self.rc,
            "step": self.step_i, "steps": self.step_n, "step_label": self.step_label,
            "elapsed": round((self.ended or time.time()) - (self.started or time.time()), 1)
            if self.started else 0,
            "queued_for": round(time.time() - self.created, 1) if self.state == "queued" else 0,
            "error": self.error,
            "next": self.seq,
            "lines": [t for s, t in self.lines if s > since],
        }


JOBS = {}
JOB_ORDER = deque(maxlen=200)
QUEUES = {"gpu": queue.Queue(), "light": queue.Queue()}
LOCK = threading.Lock()


def tree_kill(proc):
    """Kill the whole tree. segment_fossae shells out to segment_structures, so
    terminating our direct child would leave a grandchild holding the GPU."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def child_env(group=None):
    env = dict(os.environ)
    env["CT_DATA_ROOT"] = str(PROJECTS)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_step(job, caption, argv):
    job.step_label = caption
    job.emit(f"$ {' '.join(str(a) for a in argv)}")
    popen = dict(cwd=str(APP), env=child_env(), stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                 text=True, bufsize=1, errors="replace")
    if WIN:
        popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen["start_new_session"] = True
    proc = subprocess.Popen(argv, **popen)
    job.proc = proc
    for line in proc.stdout:
        job.emit(line)
        if job.cancelled:
            break
    if job.cancelled:
        tree_kill(proc)
    proc.wait()
    job.proc = None
    return proc.returncode


def worker(qname):
    q = QUEUES[qname]
    while True:
        job = q.get()
        try:
            job.state = "running"
            job.started = time.time()
            rc = 0
            for i, (caption, action) in enumerate(job.steps, 1):
                if job.cancelled:
                    break
                job.step_i = i
                if callable(action):
                    job.step_label = caption
                    job.emit(f"-- {caption}")
                    action(job)
                else:
                    rc = run_step(job, caption, action)
                    if rc != 0:
                        job.emit(f"!! exited with code {rc}")
            job.rc = rc
            job.state = ("cancelled" if job.cancelled
                         else "done" if rc == 0 else "failed")
        except Exception as e:
            traceback.print_exc()
            job.error = f"{type(e).__name__}: {e}"
            job.emit(job.error)
            job.state = "failed"
        finally:
            job.ended = time.time()
            job.step_label = ""
            try:
                d = PROJECTS / job.project / "logs" if job.project else PROJECTS / "logs"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{job.id}.log").write_text(
                    "\n".join(t for _, t in job.lines), encoding="utf-8")
            except Exception:
                pass
            q.task_done()


def submit(job):
    with LOCK:
        JOBS[job.id] = job
        JOB_ORDER.append(job.id)
    QUEUES[job.queue].put(job)
    return job


def queue_depth(qname):
    return sum(1 for j in JOBS.values()
               if j.queue == qname and j.state in ("queued", "running"))


# ------------------------------------------------------------------ job builders
PY = [sys.executable]


def job_segment(project, cases, tasks, device, license_no, force):
    """One subprocess per case per task, so progress, failure and cancellation are all
    attributable to a case rather than to a whole group."""
    steps = []
    for case in cases:
        for task in tasks:
            argv = PY + ["segment_structures.py", str(PROJECTS / project / case),
                         "--group-name", project, "--task", task,
                         "--skip-planning", "--device", device]
            if force:
                argv.append("--force-redo")
            if license_no and task in LICENSED_TASKS:
                argv += ["--license-number", license_no]
            steps.append((f"{case} - {task}", argv))
    return Job("segment", project, f"{len(tasks)} task(s) x {len(cases)} case(s)", steps)


def job_analysis(project, kind, cases, device):
    spec = ANALYSES[kind]
    steps = []
    for case in cases:
        argv = PY + [spec["script"], "--group", project, "--case", case,
                     "--device", device]
        steps.append((f"{case} - {spec['label']}", argv))
    return Job(kind, project, spec["label"], steps)


def fossa_csv(project, out):
    """The fossa table, built here rather than shelled out.

    produce_table.py only picks up stats shaped {structure: {"volume_mm3": ...}}, so it
    silently omits the fossa results, which nest under "compartments"."""
    import csv as _csv
    rows = []
    seg = seg_dir_for(project)
    for d in sorted(p for p in seg.iterdir() if p.is_dir()):
        hits = sorted(d.glob("*fossae_simple.stats.json"))
        if not hits:
            continue
        st = json.loads(hits[0].read_text())
        row = {"case": d.name, "icv_ml": st.get("icv_ml")}
        for comp, v in st.get("compartments", {}).items():
            row[f"{comp}_ml"] = v["ml"]
            row[f"{comp}_pct_icv"] = v["percent_of_icv"]
        rows.append(row)
    if not rows:
        raise RuntimeError("no fossa results yet - run the cranial fossa analysis first")
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def job_table(project, kind):
    out = seg_dir_for(project) / {
        "produce_table": f"{project}_table.csv",
        "brain_icv": "brain_icv_volumes.csv",
        "fossae": "fossa_volumes.csv"}[kind]
    if kind == "produce_table":
        step = (f"building {out.name}",
                PY + ["produce_table.py", "--group", project, "--out", str(out)])
    elif kind == "brain_icv":
        step = (f"building {out.name}",
                PY + ["brain_icv.py", "--group", project, "--out", str(out)])
    else:
        def build(job):
            n = fossa_csv(project, out)
            job.emit(f"{n} case(s) -> {out}")
        step = (f"building {out.name}", build)
    return Job("table", project, f"CSV: {kind}", [step], queue_name="light")


def job_scan(project, cases):
    """Score the DICOM series of each case, out of process so the server never imports
    torch, and so one unreadable folder cannot take the server down."""
    results = {}

    def make(case):
        def run(job):
            link = PROJECTS / project / case
            argv = PY + ["ct_gui.py", "--scan", str(link)]
            r = subprocess.run(argv, cwd=str(APP), env=child_env(),
                               capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, errors="replace")
            try:
                results[case] = json.loads(r.stdout.strip().splitlines()[-1])
                n = len(results[case].get("series", []))
                job.emit(f"{case}: {n} series, "
                         f"{results[case].get('decision', '?')}")
            except Exception:
                results[case] = {"error": (r.stderr or r.stdout or "scan failed")[-800:]}
                job.emit(f"{case}: scan failed")
        return run

    steps = [(f"scanning {c}", make(c)) for c in cases]
    job = Job("scan", project, f"scan {len(cases)} case(s)", steps, queue_name="light")
    job.results = results
    return job


# --------------------------------------------------------------- series selection
AUTO_MIN_SCORE, AUTO_MIN_GAP = 20, 15       # segment_structures.py:467-468


def scan_folder(path):
    """Run inside the --scan child. Reuses the pipeline's own scoring so the GUI can
    never disagree with what the CLI would have chosen."""
    from segment_structures import (get_series, get_series_metadata, score_series,
                                    load_cache, resolve_from_cache)
    path = Path(path)
    key = str(path.resolve())
    cache = load_cache()
    cached = None
    for k in (key, str(path)):
        if k in cache:
            files, desc, snum = resolve_from_cache(cache[k], path)
            if files:
                cached = {"snum": snum, "desc": desc,
                          "series_dir": cache[k].get("series_dir", "")}
            break

    series_map, _ = get_series(path)
    rows = []
    for uid, flist in series_map.items():
        meta = get_series_metadata(flist)
        score, reasons = score_series(meta)
        rows.append({"uid": uid, "score": score, "reasons": reasons,
                     "series_dir": str(Path(flist[0]).parent.resolve()),
                     "snum": meta["snum"], "desc": meta["desc"],
                     "slices": meta["num_slices"],
                     "thickness": meta["slice_thickness"], "kernel": meta["kernel"],
                     "axial": meta["is_axial"]})
    rows.sort(key=lambda r: -r["score"])

    decision, chosen = "none", None
    if cached:
        decision, chosen = "cached", cached
    elif rows:
        gap = rows[0]["score"] - (rows[1]["score"] if len(rows) > 1 else -999)
        if rows[0]["score"] >= AUTO_MIN_SCORE and gap >= AUTO_MIN_GAP:
            decision = "auto"
            chosen = {"snum": rows[0]["snum"], "desc": rows[0]["desc"],
                      "series_dir": rows[0]["series_dir"]}
        else:
            decision = "manual"
    return {"path": str(path), "key": key, "decision": decision, "chosen": chosen,
            "series": rows}


def write_pick(link_path, series_dir, snum, desc):
    """Record a series choice where segment_structures will find it.

    Keyed by the resolved path because Path.resolve() follows a junction, which is what
    plan_all_folders does when it builds folder_key - so the entry must live under the
    source path. The junction path is written too, which costs nothing and survives if
    that behaviour ever changes. series_dir is stored absolute; many existing entries
    are relative and only resolve when the cwd happens to be the repo root.
    """
    from segment_structures import load_cache, save_cache
    link = Path(link_path)
    entry = {"snum": str(snum), "desc": desc,
             "series_dir": str(Path(series_dir).resolve())}
    cache = load_cache()
    cache[str(link.resolve())] = entry
    cache[str(link)] = entry
    save_cache(cache)
    return entry


def clear_converted(group, case):
    """A new series pick is silently ignored unless the old NIfTI goes: the converter
    reuses any existing file with the same name (segment_structures.py:954)."""
    d = nifti_dir_for(group) / case
    n = 0
    if d.is_dir():
        for f in d.glob("*.nii.gz"):
            f.unlink()
            n += 1
    return n


# ------------------------------------------------------------------ fossa review
def _npz(group, case):
    d = seg_dir_for(group) / case
    for pat in (f"{case}_fossae_simple_curves.npz", "fossae_simple_curves.npz"):
        p = d / pat
        if p.exists():
            return p
    hits = sorted(d.glob("*fossae_simple_curves.npz"))
    return hits[0] if hits else None


def fossa_state(group, case):
    import numpy as np
    p = _npz(group, case)
    if not p:
        return None
    z = np.load(p, allow_pickle=False)
    d = seg_dir_for(group) / case
    stats = {}
    for pat in (f"{case}_fossae_simple.stats.json", "fossae_simple.stats.json"):
        if (d / pat).exists():
            stats = json.loads((d / pat).read_text())
            break
    traced = {}
    for pat in (f"{case}_fossae_simple_traced.json", "fossae_simple_traced.json"):
        if (d / pat).exists():
            traced = json.loads((d / pat).read_text())
            break
    nf = int(z["fmap"].shape[1])
    f0 = float(z["f0"])
    return {"group": group, "case": case,
            "lanes": [float(v) for v in z["l_centers"]],
            "ant": [float(v) for v in z["ant"]],
            "post": [float(v) for v in z["post"]],
            "f0": f0, "fbin": FBIN_MM, "nf": nf,
            "fmax": f0 + (nf - 1) * FBIN_MM,
            "stats": stats, "edits": traced}


def fossa_floor_png(group, case):
    """The floor map alone, one pixel per cell.

    imsave rather than a figure so pixel column i IS lane i - that exactness is what
    lets a click in the browser convert straight back to millimetres. vmin/vmax must be
    passed explicitly: cells outside the footprint are NaN, and left to work the range
    out for itself imsave masks everything and returns a blank transparent PNG with no
    error at all.
    """
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = _npz(group, case)
    if not p:
        raise RuntimeError("no floor map for this case - run the fossa analysis first")
    g = np.asarray(np.load(p, allow_pickle=False)["fmap"], float)
    if not np.isfinite(g).any():
        raise RuntimeError("the floor map is empty for this case")
    lo, hi = np.nanpercentile(g, 1), np.nanpercentile(g, 99)
    buf = io.BytesIO()
    plt.imsave(buf, np.ma.masked_invalid(np.flipud(g.T)), cmap="viridis",
               vmin=float(lo), vmax=float(hi), format="png")
    return buf.getvalue()


def job_fossa_apply(project, case, edits):
    """Save the trace, then re-run the ordinary pipeline for that case. Running the CLI
    rather than importing keeps torch out of the server; segment_fossae picks the trace
    up on its own, which is the same path an ordinary re-run takes."""
    d = seg_dir_for(project) / case
    d.mkdir(parents=True, exist_ok=True)
    tp = d / f"{case}_fossae_simple_traced.json"
    edits = {k: v for k, v in (edits or {}).items() if v}

    def write(job):
        old = d / "fossae_simple_traced.json"
        if old.exists():
            old.unlink()
        if edits:
            tp.write_text(json.dumps(edits, indent=2))
            job.emit(f"saved {', '.join(sorted(edits))} to {tp.name}")
        elif tp.exists():
            tp.unlink()
            job.emit("dropped the saved correction")

    argv = PY + ["segment_fossae.py", "--group", project, "--case", case]
    return Job("fossa_apply", project, f"{case}: apply",
               [("saving the trace", write), (f"{case} - refit", argv)])


# ---------------------------------------------------------------------- browsing
def roots():
    out = [{"name": "Home", "path": str(Path.home())}]
    if WIN:
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            p = Path(f"{letter}:\\")
            if p.exists():
                out.append({"name": f"{letter}:", "path": str(p)})
    else:
        out.append({"name": "/", "path": "/"})
    out.append({"name": "Sample scans", "path": str(APP / "ct_scans")})
    return [r for r in out if Path(r["path"]).exists()]


def browse(path):
    p = Path(path) if path else Path.home()
    if not p.is_dir():
        raise RuntimeError(f"not a folder: {p}")
    dirs = []
    try:
        for d in sorted(p.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            dirs.append({"name": d.name, "path": str(d),
                         "dicom": has_dicom_anywhere(d, budget=120)})
            if len(dirs) >= 400:
                break
    except (PermissionError, OSError) as e:
        raise RuntimeError(f"{type(e).__name__}: {e}")
    zips = sorted(f.name for f in p.glob("*.zip"))
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "roots": roots(), "dirs": dirs, "zips": zips}


# ------------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    server_version = "ct-gui"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _guard(self, q):
        """A token on every API call, and a Host check.

        Any web page the user has open can POST to 127.0.0.1, and this server exposes a
        filesystem browser and a job launcher - fossa_review.py only served precomputed
        images and did not need this.
        """
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost", "[::1]", ""):
            self._json({"error": "bad host"}, 403)
            return False
        tok = q.get("token") or self.headers.get("X-Token")
        if tok != TOKEN:
            self._json({"error": "bad or missing token"}, 403)
            return False
        return True

    def _project(self, q):
        name = q.get("project") or q.get("name") or ""
        if not load_project(name):
            raise RuntimeError(f"unknown project: {name}")
        return name

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                page = PAGE_FILE.read_text(encoding="utf-8")
                return self._send(200, page.replace("__TOKEN__", TOKEN),
                                  "text/html; charset=utf-8")
            if not u.path.startswith("/api/"):
                return self._json({"error": "not found"}, 404)
            if not self._guard(q):
                return

            if u.path == "/api/hello":
                return self._json({"app": str(APP), "projects": str(PROJECTS),
                                   "python": sys.executable})
            if u.path == "/api/tasks":
                tasks = [{"name": t, "licensed": t in LICENSED_TASKS,
                          "blurb": BLURB.get(t, "")}
                         for t in sorted(AVAILABLE_TASKS + LICENSED_TASKS)]
                return self._json({"tasks": tasks, "analyses": ANALYSES,
                                   "license_stored": license_stored()})
            if u.path == "/api/projects":
                return self._json({"projects": [
                    {"name": p["name"], "cases": len(p.get("cases", [])),
                     "source": p.get("source", ""),
                     "created": p.get("created", "")} for p in list_projects()]})
            if u.path == "/api/project":
                return self._json(project_state(self._project(q)))
            if u.path == "/api/browse":
                return self._json(browse(q.get("path")))
            if u.path == "/api/detect":
                cases, err = detect_cases(q.get("path", ""))
                return self._json({"cases": cases, "error": err,
                                   "zips": sorted(f.name for f in
                                                  Path(q.get("path", ".")).glob("*.zip"))
                                   if Path(q.get("path", ".")).is_dir() else []})
            if u.path == "/api/jobs":
                with LOCK:
                    ids = list(JOB_ORDER)[-40:]
                    js = [JOBS[i].snapshot(since=10 ** 9) for i in ids if i in JOBS]
                return self._json({"jobs": list(reversed(js)),
                                   "busy": queue_depth("gpu")})
            if u.path == "/api/job":
                jid = q.get("id", "")
                job = JOBS.get(jid)
                if not job:
                    return self._json({"error": "unknown job"}, 404)
                snap = job.snapshot(since=int(q.get("since", 0)))
                if job.kind == "scan":
                    snap["results"] = getattr(job, "results", {})
                return self._json(snap)
            if u.path == "/api/fossa/case":
                st = fossa_state(self._project(q), q.get("case", ""))
                if st is None:
                    return self._json({"error": "no fossa result for this case"}, 404)
                return self._json(st)
            if u.path == "/api/fossa/floor.png":
                png = fossa_floor_png(self._project(q), q.get("case", ""))
                return self._send(200, png, "image/png")
            if u.path == "/api/fossa/map.png":
                d = seg_dir_for(self._project(q)) / q.get("case", "")
                hits = sorted(d.glob("*fossae_simple_map.png"))
                if not hits:
                    return self._json({"error": "no map yet"}, 404)
                return self._send(200, hits[0].read_bytes(), "image/png")
            if u.path == "/api/download":
                name = self._project(q)
                f = (seg_dir_for(name) / q.get("file", "")).resolve()
                if not str(f).startswith(str(seg_dir_for(name).resolve())):
                    return self._json({"error": "outside the project"}, 400)
                if not f.is_file():
                    return self._json({"error": "not built yet"}, 404)
                return self._send(200, f.read_bytes(), "text/csv",
                                  {"Content-Disposition":
                                   f'attachment; filename="{quote(f.name)}"'})
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            q.setdefault("token", body.get("token", ""))
            if not self._guard(q):
                return

            if u.path == "/api/project/create":
                name = (body.get("name") or "").strip()
                if not SAFE_NAME.match(name):
                    return self._json({"error": "letters, numbers, spaces, - . _ only"}, 400)
                if load_project(name):
                    return self._json({"error": "a project with that name exists"}, 400)
                src = Path(body.get("source", ""))
                if not src.is_dir():
                    return self._json({"error": f"not a folder: {src}"}, 400)
                chosen = body.get("cases") or []
                cases, made, failed = [], 0, []
                for c in chosen:
                    link = PROJECTS / name / c["case"]
                    ok, msg = make_link(link, c["series_root"])
                    if ok:
                        made += 1
                        cases.append({"case": c["case"], "path": c["path"],
                                      "series_root": c["series_root"]})
                    else:
                        failed.append(f"{c['case']}: {msg}")
                if not cases:
                    return self._json({"error": "could not link any case. "
                                       + "; ".join(failed[:3])}, 400)
                save_project({"name": name, "source": str(src),
                              "created": time.strftime("%Y-%m-%d %H:%M"),
                              "cases": cases})
                return self._json({"name": name, "linked": made, "failed": failed})

            if u.path == "/api/project/delete":
                name = self._project(body)
                pr = load_project(name)
                for c in pr.get("cases", []):
                    drop_link(PROJECTS / name / c["case"])
                shutil.rmtree(PROJECTS / name, ignore_errors=True)
                return self._json({"deleted": name})

            if u.path == "/api/scan":
                name = self._project(body)
                pr = load_project(name)
                cases = body.get("cases") or [c["case"] for c in pr["cases"]]
                return self._json({"job": submit(job_scan(name, cases)).id})

            if u.path == "/api/series/pick":
                name = self._project(body)
                case = body["case"]
                entry = write_pick(PROJECTS / name / case, body["series_dir"],
                                   body["snum"], body.get("desc", ""))
                cleared = clear_converted(name, case)
                return self._json({"saved": entry, "cleared_nifti": cleared})

            if u.path == "/api/pick":
                path, why = native_folder_dialog(body.get("start") or "")
                return self._json({"path": path, "unavailable": why})

            if u.path == "/api/run":
                name = self._project(body)
                pr = load_project(name)
                cases = body.get("cases") or [c["case"] for c in pr["cases"]]
                tasks = body.get("tasks") or []
                analyses = body.get("analyses") or []
                device = body.get("device", "gpu")
                lic = (body.get("license") or "").strip()
                need_lic = [t for t in tasks if t in LICENSED_TASKS]
                if need_lic and not lic and not license_stored():
                    return self._json({"error": "these need a license number: "
                                       + ", ".join(need_lic)}, 400)
                ids = []
                if tasks:
                    ids.append(submit(job_segment(name, cases, tasks, device, lic,
                                                  bool(body.get("force")))).id)
                for a in analyses:
                    if a in ANALYSES:
                        ids.append(submit(job_analysis(name, a, cases, device)).id)
                if not ids:
                    return self._json({"error": "nothing selected"}, 400)
                return self._json({"jobs": ids})

            if u.path == "/api/table":
                name = self._project(body)
                return self._json({"job": submit(
                    job_table(name, body.get("kind", "produce_table"))).id})

            if u.path == "/api/job/cancel":
                job = JOBS.get(body.get("id", ""))
                if not job:
                    return self._json({"error": "unknown job"}, 404)
                job.cancelled = True
                tree_kill(job.proc)
                return self._json({"cancelled": job.id})

            if u.path == "/api/fossa/apply":
                name = self._project(body)
                return self._json({"job": submit(
                    job_fossa_apply(name, body["case"], body.get("edits"))).id})

            return self._json({"error": "not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


# ------------------------------------------------------------------------- modes
def selftest():
    print(f"interpreter : {sys.executable}")
    print(f"app folder  : {APP}")
    print(f"projects    : {PROJECTS}")
    for mod in ("pydicom", "dicom2nifti", "nibabel", "numpy", "matplotlib", "nrrd",
                "pandas", "totalsegmentator", "torch"):
        try:
            __import__(mod)
            print(f"  ok      {mod}")
        except Exception as e:
            print(f"  MISSING {mod}  ({type(e).__name__})")
    try:
        import torch
        print(f"cuda        : {torch.cuda.is_available()}")
    except Exception:
        pass


def free_port(start=8000, tries=12):
    for p in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise SystemExit("no free port in 8000-8011")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--open", action="store_true", help="open a browser too")
    ap.add_argument("--scan", metavar="PATH", help="internal: score a folder's series")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.scan:
        print(json.dumps(scan_folder(args.scan)))
        return
    if args.selftest:
        return selftest()

    if not PAGE_FILE.exists():
        raise SystemExit(f"missing {PAGE_FILE.name} - it must sit next to ct_gui.py")
    PROJECTS.mkdir(parents=True, exist_ok=True)
    for qname in QUEUES:
        threading.Thread(target=worker, args=(qname,), daemon=True).start()

    port = args.port or free_port()
    url = f"http://127.0.0.1:{port}/?token={TOKEN}"
    print("\n  CT segmentation - local interface")
    print(f"  open this in your browser:\n\n    {url}\n")
    print("  keep this window open while you work; ctrl-c here stops the server\n")
    if args.open:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping - killing any running job")
        for j in JOBS.values():
            if j.proc:
                j.cancelled = True
                tree_kill(j.proc)


if __name__ == "__main__":
    main()
