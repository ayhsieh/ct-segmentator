#!/bin/bash
# Double-click this once, before start.command. If this computer already has a Python
# with the segmentation packages, it uses that. Otherwise it installs a private copy of
# Python and everything the tool needs into a "miniconda" folder next to this file.
# Nothing is installed system-wide; deleting that folder undoes all of it.
cd "$(dirname "$0")"
set -u

. ./find_python.sh

PREFIX="$PWD/miniconda"
PY="$PREFIX/bin/python"

pause_and_exit() {
  echo
  read -n 1 -s -r -p "Press any key to close."
  exit "$1"
}

fail() {
  echo
  echo "  Setup failed: $1"
  echo "  Nothing was installed outside this folder. You can run this again."
  pause_and_exit 1
}

echo
echo "  Checking what this computer already has..."
echo

# Downloading several gigabytes next to an environment that already has them is the
# single worst outcome here, so every python on the machine is checked first.
if find_python && verify_python "$FOUND_PY"; then
  echo "  Everything is already installed:"
  echo "    $FOUND_PY"
  echo
  echo "  Nothing to do. Double-click start.command to begin."
  pause_and_exit 0
fi

# An environment that already has TotalSegmentator is missing only small packages -
# seconds of download against 10-30 minutes for a second copy of everything. Offered
# rather than done, because it writes into an environment used for other work.
if [ -n "$PARTIAL_PY" ]; then
  ADD=""
  for m in $PARTIAL_MISSING; do ADD="$ADD $(pip_name_for "$m")"; done
  echo "  This computer already has most of what is needed:"
  echo "    $PARTIAL_PY"
  echo "  It only needs:$ADD"
  echo
  echo "  Adding those takes about a minute. Installing a separate private copy"
  echo "  instead takes 10-30 minutes and several gigabytes of disk."
  echo
  read -r -p "  Add the missing packages to that Python? [Y/n] " ANSWER
  case "${ANSWER:-Y}" in
    [Nn]*) echo; echo "  Installing a separate copy instead." ; echo ;;
    *)
      echo
      echo "  Installing:$ADD"
      "$PARTIAL_PY" -m pip install $ADD || fail "a package would not install"
      if verify_python "$PARTIAL_PY"; then
        echo
        echo "  Done. Double-click start.command to open the tool."
        pause_and_exit 0
      fi
      echo
      echo "  That did not cover everything after all; installing a separate copy."
      echo
      ;;
  esac
fi

# 3D Slicer's TotalSegmentator extension installs into Slicer's own python, which is
# most of the way there but has no dicom2nifti or pynrrd. Adding packages to an
# application's private python can break that application, so a separate copy is
# installed instead - but say so, rather than looking like nothing was found.
for s in "/Applications/Slicer.app" "$HOME/Applications/Slicer.app"; do
  if [ -x "$s/Contents/bin/PythonSlicer" ] &&
     "$s/Contents/bin/PythonSlicer" -c "import totalsegmentator" >/dev/null 2>&1; then
    echo "  Note: 3D Slicer has TotalSegmentator, but not the other packages this needs,"
    echo "  and Slicer's own Python is left untouched. A separate copy is installed here."
    echo
  fi
done

echo "  Setting up. This downloads several gigabytes and takes 10-30 minutes."
echo "  You can leave it running. Do not close this window."
echo

# Apple silicon and Intel need different builds, and picking the wrong one fails in a
# way that is hard to read, so choose by what the machine reports.
case "$(uname -m)" in
  arm64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
  x86_64) URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
  *) fail "unsupported processor: $(uname -m)" ;;
esac

if [ ! -x "$PY" ]; then
  # A run that was interrupted mid-install leaves a half-extracted folder with no
  # python in it. -f lets the installer write over that, so running this again after a
  # closed window or a dropped connection just works.
  echo "  [1/3] Downloading Python..."
  curl -fL# "$URL" -o miniconda-installer.sh || fail "could not download Python"
  echo "  [2/3] Installing it..."
  bash miniconda-installer.sh -b -f -p "$PREFIX" >/dev/null || fail "the Python installer did not finish"
  rm -f miniconda-installer.sh
else
  echo "  [1-2/3] Python is already here, keeping it."
fi

echo "  [3/3] Installing the segmentation packages. This is the long part..."
"$PY" -m pip install --upgrade pip >/dev/null 2>&1
"$PY" -m pip install pydicom dicom2nifti nibabel numpy scipy matplotlib pandas \
      pynrrd totalsegmentator torch torchvision torchaudio || fail "a package would not install"

verify_python "$PY" || fail "TotalSegmentator did not install correctly"

echo
echo "  Done. Double-click start.command to open the tool."
pause_and_exit 0
