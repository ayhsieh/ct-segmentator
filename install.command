#!/bin/bash
# Double-click this once, before start.command. It installs a private copy of Python
# and everything the tool needs, into a "miniconda" folder next to this file. Nothing
# is installed system-wide and nothing else on the computer is touched; deleting that
# folder undoes all of it.
cd "$(dirname "$0")"
set -u

PREFIX="$PWD/miniconda"
PY="$PREFIX/bin/python"

echo
echo "  Setting up. This downloads several gigabytes and takes 10-30 minutes."
echo "  You can leave it running. Do not close this window."
echo

NEEDS="import totalsegmentator,pydicom,dicom2nifti,nibabel,scipy,matplotlib,pandas,nrrd"

if [ -x "$PY" ] && "$PY" -c "$NEEDS" >/dev/null 2>&1; then
  echo "  Already installed. Double-click start.command to begin."
  read -n 1 -s -r -p "Press any key to close."
  exit 0
fi

# 3D Slicer's TotalSegmentator extension installs into Slicer's own python, which is
# most of the way there but has no dicom2nifti or pynrrd, so it cannot simply be used
# as-is. Say so rather than silently downloading a second copy of everything.
for s in "/Applications/Slicer.app" "$HOME/Applications/Slicer.app"; do
  if [ -x "$s/Contents/bin/PythonSlicer" ] &&
     "$s/Contents/bin/PythonSlicer" -c "import totalsegmentator" >/dev/null 2>&1; then
    echo "  Note: 3D Slicer has TotalSegmentator, but not the other packages this needs,"
    echo "  so a separate copy is being installed here. Slicer is left untouched."
    echo
  fi
done

fail() {
  echo
  echo "  Setup failed: $1"
  echo "  Nothing was installed outside this folder. You can run this again."
  echo
  read -n 1 -s -r -p "Press any key to close."
  exit 1
}

# Apple silicon and Intel need different builds, and picking the wrong one fails in a
# way that is hard to read, so choose by what the machine reports.
case "$(uname -m)" in
  arm64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
  x86_64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
  *) fail "unsupported processor: $(uname -m)" ;;
esac

if [ ! -x "$PY" ]; then
  echo "  [1/3] Downloading Python..."
  curl -fL# "$URL" -o miniconda-installer.sh || fail "could not download Python"
  echo "  [2/3] Installing it..."
  bash miniconda-installer.sh -b -p "$PREFIX" >/dev/null || fail "the Python installer did not finish"
  rm -f miniconda-installer.sh
else
  echo "  [1-2/3] Python is already here, keeping it."
fi

echo "  [3/3] Installing the segmentation packages. This is the long part..."
"$PY" -m pip install --upgrade pip >/dev/null 2>&1
"$PY" -m pip install pydicom dicom2nifti nibabel numpy scipy matplotlib pandas \
      pynrrd totalsegmentator torch torchvision torchaudio || fail "a package would not install"

"$PY" -c "import totalsegmentator" >/dev/null 2>&1 || fail "TotalSegmentator did not install correctly"

echo
echo "  Done. Double-click start.command to open the tool."
echo
read -n 1 -s -r -p "Press any key to close."
