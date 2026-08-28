#!/bin/bash
# Double-click this to open the segmentation interface.
# The first time, macOS may say it is from an unidentified developer:
# right-click this file, choose Open, then Open again.
cd "$(dirname "$0")"

PY=""
for c in \
  "$HOME/anaconda3/envs/segmentator/bin/python" \
  "$HOME/miniconda3/envs/segmentator/bin/python" \
  "$HOME/opt/anaconda3/envs/segmentator/bin/python" \
  "/opt/homebrew/anaconda3/envs/segmentator/bin/python" ; do
  [ -x "$c" ] && PY="$c" && break
done

if [ -z "$PY" ]; then
  echo
  echo "  Could not find the 'segmentator' environment."
  echo "  Open Terminal and run:  conda activate segmentator"
  echo "  then:  python ct_gui.py --open"
  echo
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

echo "Starting the interface... a browser window will open."
"$PY" ct_gui.py --open
echo
read -n 1 -s -r -p "The server has stopped. Press any key to close."
