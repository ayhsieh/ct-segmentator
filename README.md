# Medical Image Segmentation

Segments anatomical structures in CT and MRI scans using
**[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** — organs, bones,
muscles, vessels, brain regions — and gives you a spreadsheet of volumes.

You drive it from your browser, but it runs entirely on your own computer. Your scans
are read where they are; nothing is uploaded or copied.

---

## Setup

Download this repository (green **Code** button → **Download ZIP**) and unzip it into
your Documents folder.

### Mac

macOS won't run downloaded scripts until you allow them. Right-click the unzipped
**folder**, hold **Option**, choose **Copy "ct-segmentator" as Pathname**. Open Terminal
(Command-Space, type Terminal), type `chmod +x ` with a space, paste, add `/*.command`,
press Return:

```bash
chmod +x /Users/you/Documents/ct-segmentator/*.command
```

Double-click **`install.command`**. macOS will call it an unidentified developer the
first time — right-click it, **Open**, **Open** again. It downloads Python and the
segmentation packages into the folder: 10–30 minutes, a few gigabytes. Do the same
right-click-Open once for **`start.command`**.

From then on, double-click **`start.command`** to use the tool. Leave its window open;
closing it stops the tool.

### Windows

Double-click **`start.bat`**. If it says nothing is installed, install
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install),
open **Anaconda Prompt**, and run:

```bash
conda create -n segmentator python=3.10 -y
conda activate segmentator
pip install pydicom dicom2nifti nibabel numpy scipy matplotlib pandas pynrrd totalsegmentator torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Drop `--index-url ...` if you have no NVIDIA graphics card.

### If nothing opens

The launcher uses the first Python it finds with all the packages, and says so if there
is none. Any Python that has them works:

```bash
python ct_gui.py --open
```

---

## Using it

**Make a project.** Click **New project** — your folder chooser opens straight away.
Pick the folder holding one subfolder per patient. Your DICOMs stay where they are;
everything generated goes into `projects/` inside the tool's folder.

> Unzip any `.zip` scans first — the pipeline deletes zip files after extracting them.

**Choose what to segment.** Search the 40 structure sets, check what you want, press
**Start segmenting**. Names ending in `_mr` are for MRI, not CT. Sets marked **license**
need a free academic number from
[here](https://backend.totalsegmentator.com/license-academic/), pasted into the box that
appears. Below the structures are two extra analyses, **cranial fossa volumes** and
**brain and intracranial volume**, both needing `brain_structures` first. Anything
already done is skipped unless you check **re-run even if done**.

**Series selection.** A scan usually holds several reconstructions of the same
acquisition — soft tissue, bone, sometimes a coronal reformat. The first time you
segment a patient the tool scores them, shows you what it picked and why, and waits for
you to confirm; where two are equally good it makes you choose. After that it remembers.
If you can't tell two apart, open the folder in 3D Slicer: you want a soft-tissue axial
reconstruction of the whole head.

**Watch it run.** Live log, progress bar, timer. Figure a minute per patient per
structure set on an NVIDIA GPU, considerably longer without one. One job runs at a time;
the next says **Queued**. Closing the browser doesn't stop it.

**Get your results.** Under **Results**, press **Build**, then **Download**:

- **All structure volumes** — every structure from every task, one row per patient
- **Brain and ICV** — parenchyma and intracranial volume
- **Cranial fossa volumes** — anterior, middle, posterior

**Show in folder** opens any of them in Finder or Explorer. Segmentations are saved
alongside as `.seg.nrrd`, which 3D Slicer opens directly.

**Correcting fossa boundaries.** Each patient with the fossa analysis gets a **review
fossa** link showing the skull floor from above and the boundaries the tool chose. Click
along where a boundary belongs and press **Apply and recompute**. Corrections survive
later runs; **Drop corrections** undoes them.

---

## From the command line

The browser tool is a front end for four scripts, which still work on their own:

```bash
python segment_structures.py --group STUDY --task brain_structures
python segment_fossae.py --group STUDY
python brain_icv.py --group STUDY
python produce_table.py --group STUDY
```

These look for study folders under `ct_scans/`; set `CT_DATA_ROOT` to keep scans
elsewhere. `python ct_gui.py --selftest` reports what is installed.
