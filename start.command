#!/bin/bash
# Double-click this to open the segmentation interface.
# The first time, macOS may say it is from an unidentified developer:
# right-click this file, choose Open, then Open again.
cd "$(dirname "$0")"
set -u

# Nothing here requires conda - any python with the packages will do, so the search
# tests candidates rather than assuming a location. See find_python.sh.
. ./find_python.sh

echo "  Looking for the Python this tool needs..."

if find_python && verify_python "$FOUND_PY"; then
  echo "  Using: $FOUND_PY"
  echo "  Starting the interface... a browser window will open."
  "$FOUND_PY" ct_gui.py --open
  echo
  read -n 1 -s -r -p "The server has stopped. Press any key to close."
  exit 0
fi

echo
if [ -n "$PARTIAL_PY" ]; then
  # Naming the environment and the handful of packages it lacks turns "nothing is
  # installed" - which is wrong and sends people into a 30 minute download - into
  # something install.command can finish in seconds.
  echo "  Almost there. This Python has most of what is needed:"
  echo "    $PARTIAL_PY"
  echo "  It is missing: $PARTIAL_MISSING"
  echo
  echo "  Double-click install.command in this folder. It will offer to add just"
  echo "  those packages, which takes about a minute."
else
  echo "  Nothing is installed yet."
  echo "  Double-click install.command in this folder first, then try again."
fi
echo
read -n 1 -s -r -p "Press any key to close."
exit 1
