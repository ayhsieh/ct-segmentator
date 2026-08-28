# Shared by install.command and start.command; not meant to be run on its own.
#
# Finding a usable Python is the same problem for both scripts, and getting it wrong
# is expensive in opposite directions: start.command giving up sends someone through
# a 10-30 minute install they did not need, and install.command not looking at all
# downloads several gigabytes next to an environment that already had the packages.
# So the search lives here once and both source it.

# The whole set the pipeline imports, not just totalsegmentator: 3D Slicer's bundled
# python and many hand-made conda envs have TotalSegmentator but no dicom2nifti or
# pynrrd, and a totalsegmentator-only test would pick a python that fails mid-run.
# Absolute, so the helper is found no matter where this is sourced from.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEEDS_MODULES="totalsegmentator pydicom dicom2nifti nibabel numpy scipy matplotlib pandas nrrd xmltodict skimage"

# pip names differ from import names in two places, so a missing module cannot simply
# be handed to pip.
pip_name_for() {
  case "$1" in
    nrrd) echo pynrrd ;;
    skimage) echo scikit-image ;;
    *) echo "$1" ;;
  esac
}

# Lists which of NEEDS_MODULES a given python lacks; empty output means all present.
missing_modules() {
  "$1" "$HERE/check_env.py" $NEEDS_MODULES 2>/dev/null
}

# Every python worth testing, best-first. Deliberately generous: conda gets installed
# under $HOME on one machine and /opt on the next, Homebrew and miniforge put it
# somewhere else again, and a person who has one of these has no reason to know which.
candidate_pythons() {
  local root env

  # The private copy this tool installs, if a previous run made one.
  echo "$PWD/miniconda/bin/python"

  # Whatever conda environment is active in this shell, and the conda that owns it.
  [ -n "${CONDA_PREFIX:-}" ] && echo "$CONDA_PREFIX/bin/python"
  [ -n "${CONDA_EXE:-}" ] && echo "$(dirname "$(dirname "$CONDA_EXE")")/bin/python"

  for root in \
    "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
    "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" \
    "/opt/miniconda3" "/opt/anaconda3" "/opt/miniforge3" \
    "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniconda/base" \
    "/usr/local/miniconda3" "/usr/local/anaconda3" ; do
    # Named environments before the base environment: someone who made an env for this
    # put the packages there, and base is the one you least want to write into.
    for env in "$root"/envs/*/bin/python; do
      [ -x "$env" ] && echo "$env"
    done
    [ -x "$root/bin/python" ] && echo "$root/bin/python"
  done

  echo "$PWD/.venv/bin/python"
  echo "$HOME/.venv/bin/python"
  echo "/Applications/Slicer.app/Contents/bin/PythonSlicer"
  echo "$HOME/Applications/Slicer.app/Contents/bin/PythonSlicer"
  command -v python3
  command -v python
}

# Sets FOUND_PY to a python that has everything, or leaves it empty. When nothing is
# complete, PARTIAL_PY/PARTIAL_MISSING describe the closest environment - one that
# already has TotalSegmentator and needs only a few small packages added. Topping that
# up is seconds of download instead of gigabytes, which is the difference between a
# person waiting and a person giving up.
find_python() {
  FOUND_PY=""; PARTIAL_PY=""; PARTIAL_MISSING=""
  local c miss seen=""
  while read -r c; do
    [ -n "$c" ] || continue
    [ -x "$c" ] || continue
    # The same interpreter turns up under several names (PATH, CONDA_PREFIX, a symlink
    # farm); resolving first keeps it from being tested and reported twice.
    c="$(cd "$(dirname "$c")" 2>/dev/null && echo "$PWD/$(basename "$c")")" || continue
    case " $seen " in *" $c "*) continue ;; esac
    seen="$seen $c"

    miss="$(missing_modules "$c")"
    if [ -z "$miss" ]; then FOUND_PY="$c"; return 0; fi

    # torch is the multi-gigabyte one and it arrives with totalsegmentator, so an env
    # that has totalsegmentator is the only kind worth topping up. Slicer's bundled
    # python is the exception: it qualifies on packages, but pip-installing into an
    # application's private interpreter can break that application, so it is only ever
    # used as-is, never modified.
    case " $miss " in *" totalsegmentator "*) continue ;; esac
    case "$c" in *Slicer.app*) continue ;; esac
    if [ -z "$PARTIAL_PY" ]; then PARTIAL_PY="$c"; PARTIAL_MISSING="$miss"; fi
  done <<EOF
$(candidate_pythons)
EOF
  return 1
}

# The real test: actually import everything, once, on the chosen python.
verify_python() {
  "$1" -c "import $(echo $NEEDS_MODULES | tr ' ' ',')" >/dev/null 2>&1
}
