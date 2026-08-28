#!/bin/bash
# Double-click this to open the segmentation interface.
# The first time, macOS may say it is from an unidentified developer:
# right-click this file, choose Open, then Open again.
cd "$(dirname "$0")"

# Conda is the documented way to install this, but nothing here requires it - any
# python that can import totalsegmentator will do, so try the conda env first and then
# whatever is on PATH. A python that cannot import it is no use: the interface would
# start and then fail on the first segmentation.
PY=""
for c in \
  "$HOME/anaconda3/envs/segmentator/bin/python" \
  "$HOME/miniconda3/envs/segmentator/bin/python" \
  "$HOME/opt/anaconda3/envs/segmentator/bin/python" \
  "/opt/homebrew/anaconda3/envs/segmentator/bin/python" \
  "$HOME/.venv/bin/python" \
  "./.venv/bin/python" \
  "$(command -v python3)" \
  "$(command -v python)" ; do
  [ -x "$c" ] || continue
  if "$c" -c "import totalsegmentator" >/dev/null 2>&1; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
  echo
  echo "  Could not find a Python with TotalSegmentator installed."
  echo
  echo "  If you followed the setup guide, open Terminal and run:"
  echo "      conda activate segmentator"
  echo "      cd \"$(pwd)\""
  echo "      python ct_gui.py --open"
  echo
  echo "  If you have not installed it yet, see the README - step 2."
  echo
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi

echo "Starting the interface... a browser window will open."
"$PY" ct_gui.py --open
echo
read -n 1 -s -r -p "The server has stopped. Press any key to close."
