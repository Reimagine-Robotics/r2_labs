# R2 coordinated release notes

Release notes for the coordinated client release — the r2_labs SDK wheel, the
VS Code extension (.vsix), and the robot backend, shipped together under one
`vX.Y.Z`. Entries are grouped by surface (SDK / Extension / Onboard / Backend).

This file is generated from changelog fragments by changie; do not edit it by
hand. Contributors record changes by adding a fragment on their PR (`changie new`
or the `/changelog` command).


## v0.10.0 - 2026-08-13
### SDK
#### Added
* robot.arm.go_to_pose(position=..., rotation=...) moves the tooltip to a cartesian pose in the world frame, taking the rotation as a [w, x, y, z] quaternion. The optional rotation_tolerance sets how far the achieved rotation may sit from the requested one before the pose is rejected as out of reach.
#### Removed
* ColumnClient.set_pwm and the COLUMN_MIN_DUTY/COLUMN_MAX_DUTY constants are removed; the column now always travels at its full-duty firmware default, with no SDK control over speed.
#### Fixed
* Current skill export supports best-effort export of a saved Uncond checkpoint while training continues, with restoration failures reported through export status.
### Backend
#### Added
* The robot resolves a cartesian tooltip pose into an arm motion, solving IK against the tool centre point from the arm's current configuration. A pose no configuration reaches, or whose rotation the closest configuration misses by more than the tolerance, is rejected before the arm moves rather than part-way through the motion.
#### Changed
* A visual trajectory motion now fails with a perception error when the reference object goes unmatched for several consecutive frames, instead of continuing to follow a trajectory it can no longer see.
* A failed behaviour now reports the option-level diagnosis (IK failure, TrajOpt infeasibility, lost visual reference) in BehaviourFailedError.error_message, where previously the SDK caller saw only the termination type.
* Enable the episode observer by default so data collection works without a per-config opt-in; recording still only starts when explicitly requested.
* Enable hue and color-temperature image augmentation by default in uncond flow-matching training (hue max-delta 0.05, color temperature 4500-8000 K)
#### Fixed
* Demo-conditioned skill training no longer fails at its first update when NNX train-state graph caching is enabled.
* Finalize training checkpoints with both a commit marker and an atomic directory rename so RPC exports cannot discover an incomplete asynchronous write

## v0.9.0 - 2026-08-11
### SDK
#### Added
* TrainerClient.train_skill_model accepts model_family="unconditioned" to train an unconditioned flow-matching skill through the learned-skills service.
#### Changed
* Learned-skills clients can train either offline model family through one endpoint and use the same current job for status, cancellation, checkpoint listing, and export.
#### Fixed
* Executing a visual trajectory via the MCP tools no longer fails with a type error.
* The offline skill-training example now uses the current TrainerClient API, defaults explicitly to unconditioned flow matching, and reports server-side training failures.
### Extension
#### Fixed
* A foot pedal that is already held down when the robot connection is established or recovers no longer registers as a press, so it can no longer advance an evaluation trial or start a recording on its own.
* A visual pose that is re-saved or replaced under an existing name now shows its current reference image in the detail dialog, instead of the previous pose's.
### Backend
#### Added
* The learned-skills service exposes standard offline unconditioned flow-matching training, with shared status, cancellation, checkpoint, and current-run export endpoints.
#### Changed
* Backend training reuses traversal of stable NNX train-state graphs to reduce host overhead and garbage-collection stalls; eager or graph-changing update functions can disable the cache.
* Visual trajectory motions use the max_linear_error and max_angular_error arguments to check the planned joint solution against the commanded pose; outside the specified threshold the motion reports a pose error instead of continuing with drifting motion.
* A foot pedal can be plugged into the machine the robot hardware is attached to, not only the machine running the backend. Setups that split the two across separate boxes can now use pedals.
* Listing the object library is much faster and transfers far less data, so the Objects view opens quickly even with a large library. Opening an object still shows its full-resolution image.
* Trajectory optimizers now accept natural-length NumPy targets and guesses through a shared interface, and greedy tracking is provided through `IKTracker`.
* Trajectory optimizers now configure joint-velocity and joint-acceleration smoothing independently; NumPy Gauss–Newton accepts guesses that violate step bounds, and `IKTracker` exposes waypoint-solve termination controls.
* `mujoco_model.KinematicModel` is now `mujoco_model.JointSpace`; NumPy Gauss-Newton trajectory optimization accepts this bounded joint domain directly, and `joint_names=None` selects every scalar, finitely limited model joint.
* Adam trajectory optimization now accepts `mujoco_model.JointSpace`, matching the Gauss-Newton configuration interface, and uses the shared MJX forward-kinematics evaluator to reduce first-solve time without changing its objective.
* Unconditioned flow-matching training avoids synchronizing accelerator state after every step, improving throughput while retaining configurable per-step ClearML loss reporting.
#### Fixed
* Synchronous skill-training startup failures now return an error response instead of escaping the RPC handler.
* Prevent excessive J4 rotation during visual trajectory replays near wrist singularities.
* Return every requested future waypoint from full-horizon visual trajectory plans.
* The notebook cell generated from a visual trajectory run no longer fails with a type error.
* Offline dataset preparation now stops as soon as its configured entry-failure limit is exceeded.
* Avoid an invalid band matrix when NumPy Gauss-Newton uses zero difference weights and no joint-step constraint.

## v0.8.0 - 2026-08-05
### SDK
#### Added
* TrainerClient.start_online_training accepts external_task_id to run online BC against an external simulation task (e.g. "maniskill/PickCube-v1_MP1000"), whose observation/action schema, cameras, and action normalization the server resolves from the task registry.
* The execution-mode query reports which modes the robot can enter, so callers can tell teleop is unavailable before requesting it.
* Set per-motion IK tolerances via max_linear_error and max_angular_error on trajectory, visual pose, and visual trajectory motion queries.
#### Fixed
* Prevent a failed RPC connection attempt from blocking unrelated SDK calls during garbage collection.
### Extension
#### Changed
* Teleop is greyed out in the mode picker on a robot with no teleop device configured, with a tooltip explaining why, instead of accepting the selection and then failing.
* Behaviour status and history update the moment they change, with less load on the robot.
#### Fixed
* Reduce robot load while a foot pedal is connected.
* Live status displays now recover on their own if updates from the robot stop arriving, instead of staying stale until you reconnect.
* Show the robot's actual execution mode in the mode picker, including modes it does not offer such as Stop.
* Deduplicate the requests a client makes when re-syncing after an events-plane interruption.
* Show readable names for visual trajectory and visual pose motions in the Behaviour tab, instead of raw type names.
### Onboard
#### Changed
* Teleop is greyed out in the mode picker, with the reason shown beneath it, on a robot with no teleop device configured — instead of accepting the tap and then failing.
### Backend
#### Added
* start_online_training accepts external_task_id: the training server configures its online-dataset exporter from the task registry preset (cameras, image size, proprio keys, dagger-split, successes-only), so external-sim (e.g. maniskill/PickCube) online-BC episodes ingest through the existing trainer.add_online_episode endpoint. The robot training path is unchanged.
* Added whole-trajectory optimisation to Visual Trajectories. This now necessitates a new service (trajopt), that must be run.
* Add an opt-in use_img_compression frontend config flag that compresses camera frames in transit — JPEG for colour, lossless 16-bit PNG for depth — cutting camera stream bandwidth. Off by default, so existing setups transport raw frames as before.
* Data Warehouse now reports SkyPilot compute that disagrees with its training-job records.
* Uncond flow-matching training takes model.dino_image_size to size the DINOv3 input, instead of always upsampling to 224x224. Default unchanged.
#### Changed
* External sim tasks train augmentation-free by default, keeping the trained policy consistent with the frames it is served.
* Visual Trajectories are now smoother due to better trajectory planning and better robot control timing.
* Visual Pose is now more accurate and moves in a linear motion from the initial pose to the target pose.
* The robot server now fails to start when its broadcast ports are unavailable, instead of starting with live camera and status displays frozen.
* Record when a live status update fails to reach connected clients, so the fault can be diagnosed instead of passing unnoticed.
* Skip random crop augmentation for cameras a dataset lacks instead of failing the training run, and crop both scene cameras by default.
* Fail a motion when the arm cannot reach its commanded poses within the configured tolerance, instead of tracking them loosely.
* Schedule blind A/B eval trials with a randomized complete block design so every model runs exactly once per round, guaranteeing balanced per-position coverage instead of only balanced totals.
#### Fixed
* Fix online BC for external sim tasks: pretrained mixing no longer crashes on a fresh growing dataset and dataset validation uses the task's proprio keys.
* Training retries now wait for the previous SkyPilot VM to be deleted before starting.

## v0.7.0 - 2026-07-24
### SDK
#### Fixed
* Raise the eval.start() timeout so starting an evaluation no longer times out.
### Extension
#### Added
* Add a "hold position until recording starts" option when recording absolute/relative trajectories, mirroring the visual trajectory workflow — the robot stays in place during setup instead of entering teach mode immediately.
#### Changed
* Visual trajectory recording now places less load on the robot server.
#### Fixed
* Preserve the recording setup form (name, type, source, and hold option) when stepping back from recording, instead of resetting it to defaults — for both trajectory and visual trajectory recording.
* Fix stale data from the previous robot after switching robots (e.g. object, trajectory, and ticket lists)
### Backend
#### Added
* Support hold-until-recording-start on the trajectory recording prepare endpoint, so the robot holds its pose during prepare instead of entering teach/teleop immediately (mirrors visual trajectory recording).
* Add an optional reset-to-home duration to collect-data/eval prepare, so callers can slow the return-home motion.
#### Changed
* Improve loading time for visual trajectory library list
* Improve loading time for visual pose library list

## v0.6.0 - 2026-07-21
### Backend
#### Changed
* Improve the gravity compensation for Piper H 1.8.6.

## v0.5.0 - 2026-07-20
### SDK
#### Added
* Record Prometheus metrics for every RPC call: request duration, server busy time, network overhead, and errors.
#### Changed
* `joint_move` example now reports measured joints and their max error from the target during the hold (instead of holding silently), with a configurable `--hold_seconds`.
### Extension
#### Changed
* The eval panel's warehouse URL prefills from the connected robot's cloud profile (via the REST `/config` endpoint) instead of a hardcoded default; an operator-edited or cleared URL is left alone.
### Backend
#### Breaking
* Refuse to start the RPC backend without an explicit system config (R2_CONFIG) and cloud profile (R2_CLOUD_PROFILE), failing with a clear error instead of silently using defaults.
#### Added
* Add online behaviour cloning — collected episodes stream live into a growing dataset that an infinite-mode flow-matching trainer consumes, publishing fresh model snapshots for hot reload; launchable via the SDK (TrainerClient.start_online_training), with the robot frontend forwarding saved episodes to the live trainer.
* The SDK artefact libraries (trajectories, visual trajectories, visual poses, objects) sync with the deployment's cloud app in the background, so an artefact taught on one robot reaches every robot and survives the loss of a compute box.
* Expose application performance metrics (control loop pacing, RPC latency, stream freshness, behaviour outcomes, visual trajectory convergence) on :9210 for site monitoring.
* The online-BC episode forwarder can drain its queued and in-flight uploads before the owning process exits (OnlineEpisodeForwarder.flush), so an online-BC run no longer loses its final episodes when the background upload worker dies with the process.
#### Changed
* Every cloud endpoint the backend touches (data warehouse, model warehouse, Sentry) resolves through the box's `R2_CLOUD_PROFILE`, and the REST `/config` endpoint reports the profile's warehouse URL for clients to default to.
* Read the data-warehouse API token from DW_API_AUTH_TOKEN on every cloud profile; the old per-profile token names are gone.
* Derive all data and library roots from R2_ROOT, so a box's .env only needs R2_ROOT instead of a dozen per-path variables.
* Declare whether a box syncs SDK artefacts to the cloud in its system config (enable_artefact_sync) instead of an env var.
#### Fixed
* SpaceNav mode transitions preserve active arm and gripper targets, and the first teleop tick holds position until a valid IK interval has elapsed.
* Keep the /health endpoint responsive when the robot backend is under heavy load, so clients no longer see spurious disconnects.
* SpaceNav teleop consistently uses the EMA joint-position controller instead of inheriting the controller selected by the previous behaviour.

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
