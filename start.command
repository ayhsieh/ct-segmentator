#!/bin/bash
# Double-click this to open the segmentation interface.
# The first time, macOS may say it is from an unidentified developer:
# right-click this file, choose Open, then Open again.
cd "$(dirname "$0")"

# Nothing here requires conda - any python with the packages will do. Candidates are
# tested rather than assumed, and against the whole set the pipeline imports, not just
# totalsegmentator: 3D Slicer's bundled python has TotalSegmentator but not dicom2nifti
# or pynrrd, so a totalsegmentator-only test would pick a python that fails later.
NEEDS="import totalsegmentator,pydicom,dicom2nifti,nibabel,scipy,matplotlib,pandas,nrrd"
PY=""
for c in \
  "./miniconda/bin/python" \
  "$HOME/anaconda3/envs/segmentator/bin/python" \
  "$HOME/miniconda3/envs/segmentator/bin/python" \
  "$HOME/opt/anaconda3/envs/segmentator/bin/python" \
  "/opt/homebrew/anaconda3/envs/segmentator/bin/python" \
  "$HOME/.venv/bin/python" \
  "./.venv/bin/python" \
  "/Applications/Slicer.app/Contents/bin/PythonSlicer" \
  "$HOME/Applications/Slicer.app/Contents/bin/PythonSlicer" \
  "$(command -v python3)" \
  "$(command -v python)" ; do
  [ -x "$c" ] || continue
  if "$c" -c "$NEEDS" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo
  echo "  Nothing is installed yet."
  echo "  Double-click install.command in this folder first, then try again."
  echo
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

echo "Starting the interface... a browser window will open."
"$PY" ct_gui.py --open
echo
read -n 1 -s -r -p "The server has stopped. Press any key to close."
