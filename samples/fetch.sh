#!/bin/bash
# Downloads the public CTs this project is tested against. Run from anywhere:
#     bash samples/fetch.sh
#
# These exist so that nothing is ever tested against a patient scan. Both are openly
# licensed and carry no patient identity; see README.md next to this file.
cd "$(dirname "$0")"
set -e

get() {
  [ -f "$1" ] && { echo "  have $1"; return; }
  echo "  fetching $1"
  curl -fL# "$2" -o "$1"
}

get head_ct_electrodes.nii.gz \
  "https://github.com/neurolabusc/niivue-images/raw/main/CT_Electrodes.nii.gz"
get torso_ct_totalseg.nii.gz \
  "https://github.com/wasserth/TotalSegmentator/raw/master/tests/reference_files/example_ct_sm.nii.gz"

echo "  done."
