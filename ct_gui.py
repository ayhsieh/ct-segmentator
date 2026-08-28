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
from collections import deque, OrderedDict
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
    "cerebral_bleed": "intracerebral hemorrhage",
    "brain_aneurysm": "intracranial aneurysms - TOF MR only, not CT",
    "ventricle_parts": "the ventricles split into their parts",
    "hip_implant": "hip prostheses",
    "pleural_pericard_effusion": "pleural and pericardial effusion",
    "liver_vessels": "hepatic vessels and tumors",
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
    "face": "the face, for defacing/anonymizing",
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

# Whether TotalSegmentator already holds a valid license. Asked once, in a subprocess,
# so the answer costs nothing at startup and torch never loads into the server. When a
# license is already stored the UI stops demanding a number for the licensed tasks -
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
r.update()                          # realise the app before asking to be brought forward

if sys.platform == "darwin":
    # -topmost raises the window, but on macOS that is not the same as activating the
    # application, and an unactivated one just bounces in the Dock: the dialog is up,
    # behind the browser, and has to be found by hand. Only the app itself can fix
    # that, by calling NSApplication activateIgnoringOtherApps:. Reached through the
    # Objective-C runtime because Tk already links AppKit and pyobjc is not a
    # dependency worth adding for one call. Doing it through AppleScript and System
    # Events works too, but triggers a one-time "wants to control System Events"
    # permission prompt, which is worse than the bug for the people this is built for.
    def _activate():
        try:
            import ctypes, ctypes.util
            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.objc_msgSend.restype = ctypes.c_void_p
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            app = objc.objc_msgSend(objc.objc_getClass(b"NSApplication"),
                                    objc.sel_registerName(b"sharedApplication"))
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            objc.objc_msgSend(app, objc.sel_registerName(b"activateIgnoringOtherApps:"),
                              True)
        except Exception:
            pass                    # worst case is the old behaviour, not a failure

    # Once now, and again once the panel is actually on screen: activating before the
    # window exists sometimes does not stick, and the retries fire inside the dialog's
    # own event loop, which is the only place left to run code while it is open.
    _activate()
    r.after(200, _activate)
    r.after(700, _activate)

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
        # no display, or a python built without Tk - the page asks for a typed path
        tail = (r.stderr or "").strip().splitlines()
        return "", (tail[-1] if tail else "no folder dialog available here")
    p = r.stdout.strip()
    return (p if p and Path(p).is_dir() else ""), ""


def reveal(target):
    """Open Explorer/Finder on target, selecting it when it is a file.

    Only ever called with a path this server built under projects/, and checked against
    that again here: an "open anything on my disk" endpoint is not what this is for.
    """
    p = Path(target).resolve()
    try:
        p.relative_to(PROJECTS.resolve())
    except ValueError:
        raise RuntimeError("that is not inside the projects folder")
    if not p.exists():
        raise RuntimeError(f"nothing at {p}")
    if WIN:
        # explorer returns 1 even when it worked, so its exit code says nothing
        subprocess.Popen(["explorer"] + (["/select,", str(p)] if p.is_file()
                                         else [str(p)]))
    elif sys.platform == "darwin":
        subprocess.Popen(["open"] + (["-R", str(p)] if p.is_file() else [str(p)]))
    else:
        subprocess.Popen(["xdg-open", str(p if p.is_dir() else p.parent)])
    return str(p)


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
    skipped = False
    for line in proc.stdout:
        job.emit(line)
        # segment_structures exits 0 after skipping a folder it could not resolve a
        # series for, which would otherwise read as a successful run that produced
        # nothing. The UI resolves series before running, so reaching this means the
        # cache changed underneath us - loud is right.
        if "requiring manual selection" in line or re.search(r"SKIP \S+: ", line):
            skipped = True
        if job.cancelled:
            break
    if job.cancelled:
        tree_kill(proc)
    proc.wait()
    job.proc = None
    if skipped and proc.returncode == 0:
        job.emit("!! this case was skipped, so nothing was recomputed - see above")
        return 65
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


def convert_case(group, case):
    """Run inside the --convert child: DICOM to NIfTI for one case.

    Out of process for the same reason --scan is, and it picks the series and names
    the file exactly the way segment_structures does, so a later segmentation run
    finds this file and reuses it rather than converting a second, differently named
    copy.
    """
    from segment_structures import (get_series, get_series_metadata, score_series,
                                    load_cache, resolve_from_cache, sanitize_filename)
    import dicom2nifti
    link = PROJECTS / group / case
    if not link.exists():
        raise RuntimeError(f"{case}: the source folder is not there")

    files = desc = None
    snum = ""
    cache = load_cache()
    for k in (str(link.resolve()), str(link)):
        if k in cache:
            files, desc, snum = resolve_from_cache(cache[k], link)
            if files:
                break
    if not files:
        # No recorded pick, so score them the way the pipeline would and take the
        # winner only if it wins clearly. Anything closer than that is a choice for a
        # person to make, on the screen that exists for it.
        series_map, _ = get_series(link)
        rows = []
        for uid, flist in series_map.items():
            meta = get_series_metadata(flist)
            score, _ = score_series(meta)
            rows.append((score, meta, flist))
        rows.sort(key=lambda r: -r[0])
        if not rows:
            raise RuntimeError(f"{case}: no DICOM series found in that folder")
        gap = rows[0][0] - (rows[1][0] if len(rows) > 1 else -999)
        if rows[0][0] < AUTO_MIN_SCORE or gap < AUTO_MIN_GAP:
            raise RuntimeError(
                f"{case}: more than one series could be the right one. Use "
                "Check series first on the project page and pick one.")
        files, desc, snum = rows[0][2], rows[0][1]["desc"], rows[0][1]["snum"]

    out = nifti_dir_for(group) / case
    out.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(f"series_{snum}_{desc}") if desc else f"series_{snum}_unnamed"
    path = out / f"{stem}.nii.gz"
    if path.exists():
        return str(path)
    dicom2nifti.dicom_series_to_nifti(str(Path(files[0]).parent), str(path))
    return str(path)


def job_convert(project, cases):
    """Convert without segmenting, so a scan can be looked at before anything is run."""
    def make(case):
        def run(job):
            argv = PY + ["ct_gui.py", "--convert", project, case]
            r = subprocess.run(argv, cwd=str(APP), env=child_env(), capture_output=True,
                               text=True, stdin=subprocess.DEVNULL, errors="replace")
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout or "conversion failed").strip()
                                   .splitlines()[-1])
            job.emit(f"{case}: converted")
        return run

    steps = [(f"converting {c}", make(c)) for c in cases]
    return Job("convert", project, f"convert {len(cases)} case(s)", steps,
               queue_name="light")


def write_pick(link_path, series_dir, snum, desc):
    """Record a series choice where segment_structures will find it.

    Keyed by the resolved path because Path.resolve() follows a junction, which is what
    plan_all_folders does when it builds folder_key - so the entry must live under the
    source path. The junction path is written too, which costs nothing and survives if
    that behavior ever changes. series_dir is stored absolute; many existing entries
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
    lets a click in the browser convert straight back to millimeters. vmin/vmax must be
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


# --------------------------------------------------------------- slice viewer
# Enough of a viewer to answer "did that segmentation work" without opening Slicer.
# The compositing happens here rather than in the browser: the volumes are already on
# this machine, numpy is quick enough at one slice, and sending a finished PNG keeps
# the page down to an <img> and a slider instead of a WebGL dependency.

# Window width and level, in HU. Bone leads because most of what this pipeline
# segments is skull, and a soft-tissue window makes a bone boundary impossible to
# judge. Sent to the page so the numbers live in exactly one place.
WINDOW_PRESETS = {"bone": [2500, 480], "soft tissue": [400, 50], "brain": [80, 40]}

PLANES = ("axial", "coronal", "sagittal")
_PLANE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}

# Volumes are slow to read and a slider asks for one slice per step, so they are held
# in memory. The key carries mtime and size, which means a pipeline re-run produces a
# new key by itself - nothing here ever has to be invalidated by hand.
_VC = OrderedDict()
_VC_LOAD = {}
_VC_LOCK = threading.Lock()     # covers _VC and _VC_LOAD only; the job LOCK is
                                # bookkeeping for something else entirely
_VC_BUDGET = int(os.environ.get("CT_VIEW_CACHE_MB", "1500")) * 1024 * 1024
_VC_ITEM_MAX = 1200 * 1024 * 1024


def _stamp(path):
    """Identity plus freshness, so a rewritten file can never be served from cache."""
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def _cached(key, loader, store=None, budget=None):
    """loader() -> (value, nbytes), called at most once per key across threads.

    store/budget pick which LRU this lives in - volumes and meshes have separate ones
    so a few dozen meshes cannot evict the CT they were made from.

    The load runs outside the lock. It can take seconds on a large volume, and holding
    the lock across it would stall every other request, including the ones that were
    about to hit the cache. A second thread asking for the same key waits on an event
    instead of reading the same file again.
    """
    store = _VC if store is None else store
    budget = _VC_BUDGET if budget is None else budget
    while True:
        with _VC_LOCK:
            if key in store:
                store.move_to_end(key)
                return store[key][0]
            ev = _VC_LOAD.get(key)
            if ev is None:
                ev = _VC_LOAD[key] = threading.Event()
                break                     # this thread owns the load
        if not ev.wait(timeout=300):
            raise RuntimeError("timed out waiting for this volume to load")
    try:
        val, n = loader()
        if n > _VC_ITEM_MAX:
            raise RuntimeError(
                f"this volume needs {n / 2 ** 30:.1f} GB of memory, which is more than "
                "this viewer will take on - open it in 3D Slicer instead")
        with _VC_LOCK:
            store[key] = (val, n)
            total = sum(v[1] for v in store.values())
            while total > budget and len(store) > 1:
                total -= store.popitem(last=False)[1][1]
        return val
    finally:
        # Set the event whatever happened. A load that failed without waking its
        # waiters would hang every later request for that file for the timeout.
        with _VC_LOCK:
            _VC_LOAD.pop(key, None)
        ev.set()


def slice2d(arr, plane, i):
    """One 2-D slice of a canonical RAS volume, oriented the way it is displayed.

    Written once and used for the CT and for every label array, so the two cannot
    drift apart by a flip. Axial and coronal are radiological - the patient's right is
    on the viewer's left - and sagittal faces left, superior up.
    """
    if plane == "axial":
        return arr[::-1, ::-1, i].T      # rows: A at top    cols: patient R at left
    if plane == "coronal":
        return arr[::-1, i, ::-1].T      # rows: S at top    cols: patient R at left
    return arr[i, ::-1, ::-1].T          # rows: S at top    cols: A at left


class NeedsConvert(RuntimeError):
    """No NIfTI for this case yet. Its own type because the page can act on it - it
    offers to convert - where an ordinary error would only be reported."""


def _ct_path(group, case):
    d = nifti_dir_for(group) / case
    hits = sorted(d.glob("*.nii.gz")) if d.is_dir() else []
    if not hits:
        raise NeedsConvert("this case has not been converted from DICOM yet")
    return hits[0]                       # the same one find_source_nifti picks


def _canonical_geometry(affine, shape, zooms):
    """Shape, zooms and affine after reorientation to RAS, without touching voxels."""
    import nibabel as nib
    ornt = nib.orientations.io_orientation(affine)
    perm = [int(a) for a in ornt[:, 0]]
    return ([int(shape[a]) for a in perm], [float(zooms[a]) for a in perm],
            affine @ nib.orientations.inv_ornt_aff(ornt, tuple(shape)))


def _ct_volume(group, case):
    """The CT in canonical RAS as int16 HU, cached."""
    import numpy as np
    import nibabel as nib
    p = _ct_path(group, case)

    def load():
        img = nib.as_closest_canonical(nib.load(str(p)))
        # int16 rather than the float the scaling produces: HU always fits, and the
        # difference is 150 MB against 300 MB for an ordinary head CT.
        vol = np.asanyarray(img.dataobj, dtype=np.int16)
        return ({"vol": vol, "aff": img.affine,
                 "zooms": [float(z) for z in img.header.get_zooms()[:3]]}, vol.nbytes)

    return _cached(("ct",) + _stamp(p), load)


def _seg_files(group, case):
    """Every .seg.nrrd for this case as (label, path), in the order to paint them.

    A glob rather than case_status(), which only reports stems that are known task
    names and so silently leaves out cranial_bones, brain_icv and the fossae file.
    Painted big to small, so total ends up underneath the structures worth seeing.
    """
    d = seg_dir_for(group) / case
    if not d.is_dir():
        return []
    rank = {"total": 0, "brain_structures": 2, "cranial_bones": 3, "brain_icv": 4}
    out = []
    for p in sorted(d.glob("*.seg.nrrd")):
        stem = p.name[: -len(".seg.nrrd")]
        label = stem[len(case) + 1:] if stem.startswith(case + "_") else stem
        out.append((5 if "fossae" in label else rank.get(label, 1), label, p))
    out.sort(key=lambda t: (t[0], t[1]))
    return [(label, p) for _, label, p in out]


def _hex_color(s):
    """A header's "r g b" floats as #rrggbb. Read rather than recomputed, so what the
    browser draws is the colour Slicer draws."""
    try:
        rgb = [float(v) for v in str(s).split()]
        if len(rgb) != 3:
            raise ValueError
    except ValueError:
        return "#c8b48c"
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v * 255)))) for v in rgb)


def _seg_segments(hdr):
    """[{i, name, value, layer, color}] from a .seg.nrrd header.

    The index is the identity used everywhere downstream - in the URL, in the colour
    table, in the DOM - because it is the one thing well defined for both file shapes
    this pipeline writes. brain_icv gives every segment LabelValue 1 and separates
    them by layer instead, so a label value on its own cannot name a segment.
    """
    def _int_or(key, default):
        try:
            return int(hdr[key])
        except (KeyError, TypeError, ValueError):
            return default

    segs, i = [], 0
    while f"Segment{i}_Name" in hdr:
        segs.append({"i": i, "name": str(hdr[f"Segment{i}_Name"]),
                     "value": _int_or(f"Segment{i}_LabelValue", i + 1),
                     "layer": _int_or(f"Segment{i}_Layer", 0),
                     "color": _hex_color(hdr.get(f"Segment{i}_Color", ""))})
        i += 1
    return segs


def _seg_affine(hdr):
    """The RAS affine and voxel shape a .seg.nrrd header describes.

    space directions and space origin are LPS, because that is what Slicer reads;
    undoing the flip multilabel_to_segnrrd applied puts it back alongside the CT's
    nibabel affine. A 4-D file carries a [nan nan nan] row for its layer axis.
    """
    import numpy as np
    dirs = np.array(hdr["space directions"], dtype=float)
    sizes = [int(v) for v in hdr["sizes"]]
    if dirs.shape[0] == 4:
        dirs, sizes = dirs[1:], sizes[1:]
    aff = np.eye(4)
    aff[:3, :3] = dirs.T
    aff[:3, 3] = np.array(hdr["space origin"], dtype=float)
    aff[0, :] *= -1
    aff[1, :] *= -1
    return aff, sizes


def _grid_matches(hdr, ct_shape, ct_aff):
    """Whether this segmentation already sits on the CT's grid - header only, no read."""
    import numpy as np
    aff, sizes = _seg_affine(hdr)
    shape, _, oaff = _canonical_geometry(aff, sizes, [1.0, 1.0, 1.0])
    return shape == list(ct_shape) and np.allclose(oaff, ct_aff, atol=1e-3)


def _seg_volume(group, case, path):
    """The segmentation as one array whose value is the segment index plus one.

    Both file shapes collapse into this. A 3-D multilabel file is remapped from label
    values to indices; brain_icv's 4-D file stacks one binary layer per segment on
    axis 0, and painting those in layer order leaves brain sitting inside ICV rather
    than one erasing the other. Everything downstream then indexes a colour table with
    the array value and needs to know nothing about which kind of file it came from.
    """
    import numpy as np
    import nibabel as nib
    import nrrd
    ct = _ct_volume(group, case)

    def load():
        data, hdr = nrrd.read(str(path), index_order="F")
        segs = _seg_segments(hdr)
        aff, _ = _seg_affine(hdr)
        layers = ([np.asarray(data[k]) for k in range(data.shape[0])]
                  if data.ndim == 4 else [np.asarray(data)])
        out = np.zeros(layers[0].shape,
                       dtype=np.uint8 if len(segs) < 255 else np.uint16)
        for s in sorted(segs, key=lambda s: s["layer"]):
            lay = layers[s["layer"]] if data.ndim == 4 else layers[0]
            out[lay == s["value"]] = s["i"] + 1
        ornt = nib.orientations.io_orientation(aff)
        oaff = aff @ nib.orientations.inv_ornt_aff(ornt, layers[0].shape)
        out = nib.orientations.apply_orientation(out, ornt)
        if out.shape != ct["vol"].shape or not np.allclose(oaff, ct["aff"], atol=1e-3):
            out = _resample_labels(out, oaff, ct)
        return {"vol": out, "segs": segs}, out.nbytes

    return _cached(("seg",) + _stamp(path) + _stamp(_ct_path(group, case)), load)


def _resample_labels(lab, laff, ct):
    """Nearest neighbour onto the CT's grid. order=0 is not a speed choice: anything
    that interpolates invents label values naming structures that were never there."""
    import numpy as np
    from scipy import ndimage
    try:
        m = np.linalg.inv(laff) @ ct["aff"]
    except np.linalg.LinAlgError:
        raise RuntimeError("this segmentation's grid cannot be lined up with the CT")
    if not np.all(np.isfinite(m)):
        raise RuntimeError("this segmentation's grid cannot be lined up with the CT")
    return ndimage.affine_transform(lab, m[:3, :3], offset=m[:3, 3],
                                    output_shape=ct["vol"].shape, order=0,
                                    mode="constant", cval=0,
                                    prefilter=False).astype(lab.dtype)


def view_case(group, case):
    """Everything the viewer needs to draw its chrome, without reading a voxel.

    nibabel loads lazily and pynrrd will read a header on its own, so this stays in
    milliseconds even for total, whose labels are 80 MB. The page is therefore up and
    interactive before the first slice has been asked for.
    """
    import nibabel as nib
    import nrrd
    p = _ct_path(group, case)
    img = nib.load(str(p))
    shape, zooms, aff = _canonical_geometry(img.affine, img.shape[:3],
                                            img.header.get_zooms()[:3])
    (nx, ny, nz), (zx, zy, zz) = shape, zooms
    planes = {
        "axial": {"n": nz, "w": nx, "h": ny, "mm_w": nx * zx, "mm_h": ny * zy,
                  "dir": "I → S"},
        "coronal": {"n": ny, "w": nx, "h": nz, "mm_w": nx * zx, "mm_h": nz * zz,
                    "dir": "P → A"},
        "sagittal": {"n": nx, "w": ny, "h": nz, "mm_w": ny * zy, "mm_h": nz * zz,
                     "dir": "L → R"},
    }
    layers, warnings, stamps = [], [], [p.stat().st_mtime_ns]
    for label, sp in _seg_files(group, case):
        stamps.append(sp.stat().st_mtime_ns)
        try:
            hdr = nrrd.read_header(str(sp))
            segs = _seg_segments(hdr)
            status = "ok" if _grid_matches(hdr, shape, aff) else "resample"
        except Exception as e:
            # One unreadable file must not cost you the rest of the case.
            warnings.append(f"{sp.name} could not be read: {e}")
            continue
        if not segs:
            continue
        if status == "resample":
            warnings.append(f"{label} sits on a different grid to the CT, so it is "
                            "resampled to match - boundaries may be a voxel out")
        layers.append({"task": label, "status": status,
                       "segments": [{"i": s["i"], "name": s["name"],
                                     "color": s["color"]} for s in segs]})
    return {"project": group, "case": case, "shape": shape, "zooms": zooms,
            "planes": planes, "presets": WINDOW_PRESETS, "default": "bone",
            "stamp": max(stamps), "layers": layers, "warnings": warnings}


def _qint(q, key, default):
    """A query integer that falls back rather than 500s - these arrive from a slider
    and a drag, and a half-typed value is not worth an error page."""
    try:
        return int(float(q.get(key, "")))
    except (TypeError, ValueError):
        return default


def _parse_visible(s):
    """"total:1f0a,brain_structures:ff03" -> {task: bitset over segment index}.

    A bitset rather than a list of names because total has 117 structures: 15 bytes of
    hex against a kilometre of URL, and it keeps the address a stable cache key, which
    is what lets the browser re-show a slice it has already seen without asking.
    """
    out = {}
    for part in (s or "").split(","):
        name, sep, bits = part.partition(":")
        if not sep:
            continue
        try:
            out[name] = bytes.fromhex(bits)
        except ValueError:
            continue
    return out


def view_slice_png(group, case, plane, i, ww, wl, op, vis):
    """One composited slice as PNG bytes: CT windowed to grey, structures blended on."""
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ct = _ct_volume(group, case)
    n = ct["vol"].shape[_PLANE_AXIS[plane]]
    i = max(0, min(n - 1, i))            # clamp rather than fail: a stale index from a
    sl = slice2d(ct["vol"], plane, i)    # re-fetched case must not break the view
    ww = max(1.0, float(ww))
    lo = float(wl) - ww / 2.0
    g = np.clip((sl.astype(np.float32) - lo) * (255.0 / ww), 0, 255).astype(np.uint8)
    rgb = np.repeat(g[:, :, None], 3, axis=2)

    a = max(0.0, min(1.0, op / 100.0))
    for label, sp in (_seg_files(group, case) if a > 0 else []):
        want = vis.get(label)
        if not want or not any(want):
            continue
        try:
            seg = _seg_volume(group, case, sp)
        except Exception:
            traceback.print_exc()        # one bad layer, not a blank viewer
            continue
        keep = np.zeros(len(seg["segs"]) + 1, dtype=bool)
        lut = np.zeros((len(seg["segs"]) + 1, 3), dtype=np.uint8)
        for s in seg["segs"]:
            byte = s["i"] // 8
            if byte < len(want) and (want[byte] >> (s["i"] % 8)) & 1:
                keep[s["i"] + 1] = True
            c = s["color"].lstrip("#")
            lut[s["i"] + 1] = [int(c[k:k + 2], 16) for k in (0, 2, 4)]
        lab = slice2d(seg["vol"], plane, i)
        m = keep[lab]
        if m.any():
            # Fancy-indexed so the work tracks the painted area, not the frame.
            rgb[m] = (rgb[m] * (1 - a) + lut[lab[m]] * a).astype(np.uint8)

    buf = io.BytesIO()
    plt.imsave(buf, rgb, format="png")   # 3-channel uint8 passes straight through
    return buf.getvalue()


# ------------------------------------------------------------------ 3d surfaces
# Scrolling slices tells you whether a boundary is right. It does not tell you whether
# a structure came out the right shape - a segmentation that leaked into the next lobe
# is obvious the moment you spin it and easy to miss slice by slice. So: marching cubes
# here, a small WebGL viewer in the page, and nothing else. Not a replacement for
# Slicer, which is still where you go to fix anything.

MESH_MAGIC = b"MSH1"
MESH_MAX_TRIS = 1_200_000

# Meshes get their own budget rather than sharing the volume cache. A mesh is a couple
# of megabytes and a CT is two hundred, so a few dozen meshes in one LRU would push the
# CT out and the next slice would re-read it from disk - a viewer that gets slower the
# more you use it.
_MC = OrderedDict()
_MC_BUDGET = int(os.environ.get("CT_MESH_CACHE_MB", "256")) * 1024 * 1024


def _pool(a, d):
    """Downsample a boolean mask to fractional occupancy by block mean.

    Not point-sampling, which is what a[::2] and marching_cubes(step_size=2) both do:
    a skull table one voxel thick falls between the samples and vanishes. Averaging
    keeps it as partial occupancy, and being a box prefilter it also comes out smoother
    than marching a binary mask, so no separate smoothing pass is needed.
    """
    import numpy as np
    if d == 1:
        return a.astype(np.float32)
    s = [n - n % d for n in a.shape]
    b = a[:s[0], :s[1], :s[2]].astype(np.float32)
    return b.reshape(s[0] // d, d, s[1] // d, d, s[2] // d, d).mean(axis=(1, 3, 5))


def _pack_mesh(verts, normals, faces, d):
    """The wire format, gzipped:

        "MSH1" | nverts u32 | ntris u32 | d u32 | bbox_min 3f | bbox_max 3f
              | verts f32[n*3] | normals f32[n*3] | faces u32[t*3]

    Binary because a structure is tens of thousands of triangles and the same thing as
    JSON would be several times the bytes and a few hundred milliseconds of parsing.
    The bounding box rides in the header so the page can frame the camera without
    scanning the vertices. gzip level 1 is about four times smaller for five
    milliseconds, which is the whole bandwidth question answered without quantising.
    """
    import gzip
    import struct
    import numpy as np
    if len(verts):
        lo, hi = verts.min(0), verts.max(0)
    else:
        lo = hi = np.zeros(3, np.float32)
    head = struct.pack("<4sIII6f", MESH_MAGIC, len(verts), len(faces), d,
                       float(lo[0]), float(lo[1]), float(lo[2]),
                       float(hi[0]), float(hi[1]), float(hi[2]))
    return gzip.compress(head + verts.astype(np.float32).tobytes()
                         + normals.astype(np.float32).tobytes()
                         + faces.astype(np.uint32).tobytes(), 1)


def view_mesh(group, case, task, index):
    """One structure's surface in millimetres, as a gzipped binary blob.

    One structure per request rather than all the visible ones together: the page can
    draw the first surface while the rest are still being built, toggling one eye does
    not re-transfer the others, and a structure that fails costs only itself.
    """
    import numpy as np
    from scipy import ndimage
    from skimage import measure

    hit = [p for label, p in _seg_files(group, case) if label == task]
    if not hit:
        raise RuntimeError(f"no segmentation called {task!r} for this case")
    path = hit[0]

    def load():
        seg = _seg_volume(group, case, path)
        if not any(s["i"] == index for s in seg["segs"]):
            raise RuntimeError(f"{task} has no structure {index}")
        zx, zy, zz = _ct_volume(group, case)["zooms"]
        m = seg["vol"] == index + 1
        empty = (np.zeros((0, 3), np.float32),) * 2 + (np.zeros((0, 3), np.uint32),)
        if not m.any():
            # A task can legitimately contain a structure this patient does not have.
            blob = _pack_mesh(*empty, 1)
            return blob, len(blob)

        # Crop to the structure before doing anything expensive. A brain occupies
        # maybe an eighth of the scan, and marching cubes costs what you hand it.
        ax = [np.flatnonzero(m.any(axis=(1, 2))), np.flatnonzero(m.any(axis=(0, 2))),
              np.flatnonzero(m.any(axis=(0, 1)))]
        lo = [int(a[0]) for a in ax]
        sub = m[lo[0]:int(ax[0][-1]) + 1, lo[1]:int(ax[1][-1]) + 1,
                lo[2]:int(ax[2][-1]) + 1]
        d = 1 if sub.size <= 8_000_000 else 2 if sub.size <= 64_000_000 else 3
        sp = np.array([zx * d, zy * d, zz * d], dtype=np.float32)

        # Blur the mask into fractional occupancy before marching. Marching a boolean
        # straight gives a terraced surface - you see the slice thickness as rings,
        # which reads as anatomy and is not. Sigma is set in millimetres and converted
        # per axis, so a scan with 5 mm slices gets the smoothing it actually needs
        # along z rather than the same voxel count as a 0.4 mm axis.
        sig_mm = 0.8 * max(zx, zy, zz)
        occ = ndimage.gaussian_filter(_pool(sub, d), 
                                      sigma=[sig_mm / (zx * d), sig_mm / (zy * d),
                                             sig_mm / (zz * d)], mode="constant")
        # The zero pad closes anything running off the edge of the scan; without it a
        # structure the field of view clipped renders as an open shell you see into.
        # level below 0.5 because blurring pulls a thin sheet's peak down, and losing
        # a one-voxel bone table matters more than half a voxel of dilation.
        try:
            v, f, n, _ = measure.marching_cubes(np.pad(occ, 1), level=0.42,
                                                spacing=tuple(sp), step_size=1,
                                                allow_degenerate=False)
        except (ValueError, RuntimeError):
            blob = _pack_mesh(*empty, d)
            return blob, len(blob)
        if len(f) > MESH_MAX_TRIS:
            raise RuntimeError("this structure is too detailed to show in 3D - "
                               "open the .seg.nrrd in 3D Slicer instead")
        v = v.astype(np.float32) - sp                      # undo the pad
        v += np.array([lo[0] * zx, lo[1] * zy, lo[2] * zz], dtype=np.float32)
        blob = _pack_mesh(v, n, f, d)
        return blob, len(blob)

    return _cached(("mesh",) + _stamp(path) + _stamp(_ct_path(group, case)) + (index,),
                   load, store=_MC, budget=_MC_BUDGET)


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
        # no-store unless the caller asked for something else - slice images are
        # addressed by a URL carrying the file's mtime, so they are safe to cache and
        # scrubbing back over slices already seen should not hit the server at all.
        if not any(k.lower() == "cache-control" for k in (extra or {})):
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

        Any web page the user has open can POST to 127.0.0.1, and this server opens
        file dialogs and launches jobs - fossa_review.py only served precomputed images
        and did not need this.
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

    def _case(self, q):
        """A case name that is a name, not a path. Everything built from it is joined
        onto the project's own folders, so a stray slash must not get through."""
        case = q.get("case", "")
        if not SAFE_NAME.match(case):
            raise RuntimeError(f"not a case name: {case!r}")
        return case

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
                     "description": p.get("description", ""),
                     "created": p.get("created", "")} for p in list_projects()]})
            if u.path == "/api/project":
                return self._json(project_state(self._project(q)))
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
            if u.path == "/api/view/case":
                try:
                    return self._json(view_case(self._project(q), self._case(q)))
                except NeedsConvert as e:
                    # Not an error so much as a next step, and the page offers it.
                    return self._json({"error": str(e), "needs_convert": True}, 409)
            if u.path == "/api/view/slice.png":
                plane = q.get("plane", "axial")
                png = view_slice_png(
                    self._project(q), self._case(q),
                    plane if plane in PLANES else "axial",
                    _qint(q, "i", 0), _qint(q, "ww", 2500), _qint(q, "wl", 480),
                    _qint(q, "op", 45), _parse_visible(q.get("v", "")))
                # Safe to cache: the URL carries the files' mtime, so a re-run gives a
                # different address. This is what makes scrubbing back free.
                return self._send(200, png, "image/png",
                                  {"Cache-Control": "private, max-age=300"})
            if u.path == "/api/view/mesh.bin":
                blob = view_mesh(self._project(q), self._case(q),
                                 q.get("task", ""), _qint(q, "i", -1))
                # Already gzipped by view_mesh, so the browser is told to inflate it.
                return self._send(200, blob, "application/octet-stream",
                                  {"Cache-Control": "private, max-age=300",
                                   "Content-Encoding": "gzip"})
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
                              "description": (body.get("description") or "").strip()[:2000],
                              "cases": cases})
                return self._json({"name": name, "linked": made, "failed": failed})

            if u.path == "/api/project/describe":
                name = self._project(body)
                pr = load_project(name)
                pr["description"] = (body.get("description") or "").strip()[:2000]
                save_project(pr)
                return self._json({"description": pr["description"]})

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

            if u.path == "/api/convert":
                name = self._project(body)
                case = body.get("case", "")
                if not SAFE_NAME.match(case):
                    return self._json({"error": f"not a case name: {case!r}"}, 400)
                return self._json({"job": submit(job_convert(name, [case])).id})

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

            if u.path == "/api/reveal":
                name = self._project(body)
                base = seg_dir_for(name)
                case, f = body.get("case") or "", body.get("file") or ""
                target = base / case / f if case else base / f
                return self._json({"opened": reveal(target)})

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
    # xmltodict is in here because TotalSegmentator imports it but does not require
    # it: without it a segmentation runs to the end and then writes no .seg.nrrd.
    for mod in ("pydicom", "dicom2nifti", "nibabel", "numpy", "matplotlib", "nrrd",
                "pandas", "xmltodict", "skimage", "totalsegmentator", "torch"):
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
    ap.add_argument("--convert", nargs=2, metavar=("GROUP", "CASE"),
                    help="internal: convert one case from DICOM to NIfTI")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.scan:
        print(json.dumps(scan_folder(args.scan)))
        return
    if args.convert:
        print(convert_case(*args.convert))
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
