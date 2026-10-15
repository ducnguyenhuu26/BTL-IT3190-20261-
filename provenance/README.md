# Historical source snapshots

`ede2f98b9ae4/` contains the exact original source bytes associated with the
USGS ablation lock SHA-256 beginning `ede2f98b9ae4`. Its manifest records the
original source checksums and extra reference files used for optimization
regression tests. These are source files, not datasets or trained models.

The working implementation is under `src/` and `scripts/`. Historical copies
are kept here because intentionally optimizing those files changes their hashes.
Report-only audits can verify the original snapshot together with the unchanged
models, predictions, and validation lock. Prediction/test stages still require
the current source to match the original lock and never use this fallback.

To reproduce a historical implementation, restore its source files into a
separate checkout with the matching local data, models and environment. Do not
reset a completed experiment's test marker or replace its lock. To run a new
implementation, use a new output directory and a new validation lock.

Git attributes disable newline conversion for this directory to preserve the
archived byte checksums on both Windows and Linux.
