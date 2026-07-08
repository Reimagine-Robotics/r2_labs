# R2 coordinated release notes

Release notes for the coordinated client release — the r2_labs SDK wheel, the
VS Code extension (.vsix), and the robot backend, shipped together under one
`vX.Y.Z`. Entries are grouped by surface (SDK / Extension / Onboard / Backend).

This file is generated from changelog fragments by changie; do not edit it by
hand. Contributors record changes by adding a fragment on their PR (`changie new`
or the `/changelog` command).


## v0.4.0 - 2026-07-08
### SDK
#### Breaking
* Remove `episode_prefix` from `EvalConfigQuery`; eval episodes always save under a fixed `eval_{task}` prefix.
* `EvalConfigQuery` requires a `location` and accepts optional `tags`; eval episodes save under `eval_{location}_{task}` and task/location are validated against the warehouse at configure time.
#### Removed
* `r2_labs.sdk.logging.configure` no longer writes rotating log files to `R2_LOG_DIR` (default `/var/log/r2`); services log to stderr and container logs are shipped centrally.
### Extension
#### Changed
* Drop the episode-prefix field from the eval panel; a task name is required to start a session and episodes always save.
* Stream Collect Data camera previews over MJPEG instead of polling frames
* The eval panel's free-text Task input becomes a dropdown of warehouse-defined tasks with a guided flow for adding new ones, alongside a Location dropdown and a session tags input that surfaces existing tags.
#### Fixed
* Fix trajectory visualization timing out in the IDE after the robot's backend is restarted; the viewer now re-establishes its connection when you reconnect to the robot.
* Keep live camera feeds streaming after repeatedly switching between tabs or refocusing the window, instead of freezing on "Loading camera...".
* After an eval session completes, Upload is the primary action (starting over is confirmed while results are un-uploaded), a finished upload links to the warehouse session, and the operator field suggests known names from the warehouse.
### Backend
#### Added
* Link each eval trial to its recorded episode in the data warehouse, and push eval episodes to the cloud warehouse automatically after a session uploads.
* Saved episodes (data collection and evals) carry a firmware:<version> tag read from the arms at save time, plus a joint_positions_controller:<description> tag with the running controller's type and key parameters (gains, limits, ki), so datasets can be filtered by the robot software that produced them.
* Canonical task and location vocabularies for eval sessions. Prod rows are migrated, uploads are validated, clients read values from GET /api/eval/enums/, new tasks are added via POST /api/eval/tasks/, sessions carry reusable tags, and the eval stats endpoints take a location filter.
* Select which on-robot system config runs by name (R2_CONFIG), so operators can switch between shipped config variants — including whether teleop (gello) and cuff-button controls are enabled — without editing source.
#### Changed
* Behaviours now run under the jerk-limited Ruckig joint position controller by default; learned behaviours run under EMA to match the controller their training data was recorded with. The controller is rebuilt only when the requirement changes between behaviours, and operator modes (teach, teleop) keep using the configured type.
* Lower the default camera web-stream frame rate from 20 to 10 fps.
#### Fixed
* Fix execution-mode and cuff-press state going stale in the extension and onboard views instead of updating live.
* Eval UX fixes from the workflow review — the sessions table's task and location columns no longer overlap, leaderboard ranks only show within a task partition, compare selections are shareable URLs with A/B mutually excluded and errors explained, and session detail reconciles machine and hardware info.

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
