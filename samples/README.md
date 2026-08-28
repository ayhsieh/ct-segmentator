# Sample CTs

Public, openly licensed scans for testing. **Nothing in this project should be tested
against a patient scan** - not rendered, not screenshotted, not opened in a viewer.
These are here so there is never a reason to.

The files themselves are not committed; `fetch.sh` downloads them.

```bash
bash samples/fetch.sh
```

| file | what it is | source | licence |
|---|---|---|---|
| `head_ct_electrodes.nii.gz` | head CT with implanted electrodes, 160 x 232 x 160 at ~1 mm | [niivue-images](https://github.com/neurolabusc/niivue-images), from the [SCI Institute](https://www.sci.utah.edu/) Seg3D data | MIT |
| `torso_ct_totalseg.nii.gz` | downsampled torso CT, 3 mm, TotalSegmentator's own test input | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) `tests/reference_files/example_ct_sm.nii.gz` | Apache-2.0 / CC BY-4.0 dataset |

Use the head CT for anything cranial - `brain_structures`, `cranial_bones`, the fossa
analysis - and the torso one for `total`, where its 3 mm voxels keep a run to seconds.

## Synthetic fixtures

`make_fixtures.py` builds two projects of shapes with geometry you can assert on, which
catches things a real scan cannot - deliberately anisotropic voxels and a non-RAS
affine turn orientation and aspect bugs into wrong numbers rather than something you
have to eyeball.

```bash
python samples/make_fixtures.py
```

`SHAPES_TEST` is a sphere, a cube, a slab running off the edge of the volume, and a
nested ICV/brain pair written as the 4-D two-layer `.seg.nrrd` that `brain_icv`
produces. `MANY_TEST` is sixty blobs, for the 3D display cap and for timing. Delete
both from the project list when you are done.
