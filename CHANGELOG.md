# R2 coordinated release notes

Release notes for the coordinated client release — the r2_labs SDK wheel, the
VS Code extension (.vsix), and the robot backend, shipped together under one
`vX.Y.Z`. Entries are grouped by surface (SDK / Extension / Onboard / Backend).

This file is generated from changelog fragments by changie; do not edit it by
hand. Contributors record changes by adding a fragment on their PR (`changie new`
or the `/changelog` command).


## v0.3.0 - 2026-07-02
### SDK
#### Added
* Train from a pre-staged dataset by setting `dataset_cache_key` on a skill training query.
### Extension
#### Changed
* Trajectory recording status now updates in real time via server push instead of polling.

## v0.2.0 - 2026-06-22

First coordinated client release: the r2_labs wheel, the VS Code extension, and
the backend cut together under one version. This entry is the baseline for the
fragment-based release notes; later versions are generated from PR fragments.
