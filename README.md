# Medical Image Segmentation

Segments anatomical structures in CT and MRI scans using
**[TotalSegmentator](https://github.com/wasserth/TotalSegmentator)** — organs, bones,
muscles, vessels, brain regions — and gives you a spreadsheet of volumes.

You drive it from your web browser, but it runs entirely on your own computer. Your
scans are read where they are; nothing is uploaded or copied.

---

## Setup

Download this repository (green **Code** button → **Download ZIP**) and unzip it into
your Documents folder.

<p align="center">
  <img src="tutorial_images/download_button.png" />
</p>

### Mac

macOS won't run downloaded scripts until you allow them. Right-click the unzipped
**folder**, hold **Option**, choose **Copy "ct-segmentator" as Pathname**. Open Terminal
(Command-Space, type Terminal), type `chmod +x ` with a space, paste, add `/*.command`,
and press Return:

```bash
chmod +x /Users/you/Documents/ct-segmentator/*.command
```

Now double-click **`install.command`**. The first time, macOS says it's from an
unidentified developer — right-click it, choose **Open**, then **Open** again. It
downloads Python and the segmentation packages into the folder, which takes 10–30
minutes and a few gigabytes. Do the same right-click-Open once for **`start.command`**.

After that, double-click **`start.command`** whenever you want the tool. Leave its window
open while you work; closing it stops the tool.

### Windows

Double-click **`start.bat`**. If it says nothing is installed, you need Python first:
install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install),
open **Anaconda Prompt**, and run:

```bash
conda create -n segmentator python=3.10 -y
conda activate segmentator
pip install pydicom dicom2nifti nibabel numpy scipy matplotlib pandas pynrrd totalsegmentator torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Drop `--index-url ...` if you don't have an NVIDIA graphics card.

### If nothing opens

The launcher uses the first Python it finds with all the packages: the `miniconda` folder
`install.command` creates, then a `segmentator` conda environment, then 3D Slicer's, then
`python` on your PATH. Slicer's own TotalSegmentator install is normally skipped — it has
TotalSegmentator but not the DICOM packages — so `install.command` is still needed. If
nothing qualifies, the window says so. Any Python that does will work:

```bash
python ct_gui.py --open
```

---

## Using it

### Make a project

A project points at a folder of scans. Click **New project** — your folder chooser opens
straight away. Pick the folder holding one subfolder per patient, and it lists what it
found before creating anything. Your DICOMs stay where they are; everything generated
goes into `projects/` inside the tool's folder.

> Unzip any `.zip` scans first. The tool warns you if it sees any, because the pipeline
> deletes zip files after extracting them.

### Choose what to segment

Search the 40 structure sets, check what you want, press **Start segmenting**. Names
ending in `_mr` are for MRI, not CT.

Sets marked **license** need a free academic license number, which you can request
[here](https://backend.totalsegmentator.com/license-academic/). Paste it in the box that
appears — if TotalSegmentator already has one, no box appears.

Below the structures are two extra analyses, **cranial fossa volumes** and **brain and
intracranial volume**. Both need `brain_structures` first.

Anything already done is marked and skipped unless you check **re-run even if done**.

### Watch it run

Live log, progress bar, timer. Figure a minute per patient per structure set on an NVIDIA
GPU, considerably longer without one. One job runs at a time; the next says **Queued**.
Closing the browser doesn't stop it.

### Check the series (optional)

A scan usually holds several reconstructions of the same acquisition — soft tissue, bone,
sometimes a coronal reformat. The tool scores them and picks one, and is right almost
always. **Check series first** shows the ranking and its reasoning, and asks you to
decide when two are close. If you can't tell them apart, open the folder in 3D Slicer:
you want a soft-tissue axial reconstruction of the whole head.

### Get your results

Under **Results**, press **Build**, then **Download**:

- **All structure volumes** — every structure from every task, one row per patient
- **Brain and ICV** — parenchyma and intracranial volume
- **Cranial fossa volumes** — anterior, middle, posterior

**Show in folder** opens any of them in Finder or Explorer. Segmentations are saved
alongside as `.seg.nrrd`, which 3D Slicer opens directly.

### Adding more later

Reopen the project and check another structure set. Only the new one runs.

### Correcting fossa boundaries

Each patient that has the fossa analysis gets a **review fossa** link, showing the skull
floor from above with the boundaries the tool chose. If one is wrong, click along where
it belongs and press **Apply and recompute**. Corrections survive later runs; **Drop
corrections** undoes them.

---

## Using it from the command line

The browser tool is a front end for four scripts, which still work on their own:

```bash
python segment_structures.py --group STUDY --task brain_structures
python segment_fossae.py --group STUDY
python brain_icv.py --group STUDY
python produce_table.py --group STUDY
```

By default these look for study folders under `ct_scans/`. Set `CT_DATA_ROOT` to keep
your scans somewhere else.

To check your installation:

```bash
python ct_gui.py --selftest
```
