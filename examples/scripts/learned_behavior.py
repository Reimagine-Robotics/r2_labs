"""CLI for executing learned behaviors via local or remote inference.

Simple usage (default):
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160"

Workflow:
  - Starts in teleop (gello) mode
  - Press Enter to start the policy
  - Press Enter to stop and align leader arm
  - Press Enter again to enable teleop control
  - Press Enter to resume the policy
  - Ctrl+C to quit

With progress-based termination (local model):
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160" \
    --termination_model_id="progress_prediction#1" \
    --termination_threshold=0.95

With progress-based termination (remote service):
  # First, start the termination model server:
  uv run python -m bot01.inference.service.run_serve_stablehlo_model \
    --cfg.model_id="progress_prediction#1" \
    --cfg.service_port=4244

  # Then run with remote termination:
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160" \
    --termination_service_address="tcp://localhost:4244" \
    --termination_threshold=0.95

DAgger mode (with pedal + episode recording) and preloading models:
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160" \
    --enable_dagger \
    --entry_prefix=dagger_dcam \
    --preload_models

DAgger workflow:
  - Press pedal A to begin (reset pose -> align -> teleop)
  - Press pedal A to start policy (recording begins here)
  - Press pedal A to stop policy and align leader arm
  - Press pedal A again to enable teleop control
  - Press pedal B to save episode
  - Press pedal C to discard episode
  - Ctrl+C to quit

Pedal controls (DAgger mode):
  - A (start/resume): Start episode or resume policy
  - A (during policy): Stop and align leader arm
  - A (after alignment): Enable teleop control
  - B: Save current episode
  - C: Discard current episode
"""

from __future__ import annotations

import enum
import select
import signal
import sys
import threading
import time

import dotenv
import evdev
from absl import app, flags
from evdev import InputDevice, ecodes
from loguru import logger as log

from r2_labs import client as r2client
from r2_labs import rpc_api
from r2_labs.sdk import futures
from r2_labs.sdk import logging as r2_logging
from r2_labs.sdk import sentry

FLAGS = flags.FLAGS

# Model configuration
flags.DEFINE_string(
    "model_id",
    "",
    "Model ID for local inference",
)
flags.DEFINE_string(
    "service_address",
    "",
    "Service address for remote inference (e.g. tcp://gpu-machine:4243)",
)
flags.DEFINE_float(
    "timeout",
    None,
    "Timeout in seconds for policy execution",
)
flags.DEFINE_integer(
    "obs_history_len",
    1,
    "Number of past observation timesteps to provide to model",
)
flags.DEFINE_integer(
    "buffer_actions",
    20,
    "Number of actions to buffer before rerunning inference",
)
flags.DEFINE_integer(
    "action_offset",
    2,
    "Number of action timesteps to offset by",
)
flags.DEFINE_string(
    "action_key",
    "action",
    "Key for action in model output",
)
flags.DEFINE_string(
    "server",
    "localhost",
    "Robot server hostname",
)

# Termination model flags
flags.DEFINE_string(
    "termination_model_id",
    "",
    "Model ID for local termination prediction",
)
flags.DEFINE_string(
    "termination_service_address",
    "",
    "Service address for remote termination model (e.g. tcp://gpu-machine:4244)",
)
flags.DEFINE_float(
    "termination_threshold",
    0.95,
    "Progress threshold for termination",
)
flags.DEFINE_integer(
    "termination_min_frames",
    2,
    "Consecutive frames above threshold before terminating",
)
flags.DEFINE_float(
    "poll_interval",
    0.1,
    "Interval between termination checks (seconds)",
)

# DAgger mode
flags.DEFINE_bool(
    "enable_dagger",
    False,
    "Enable DAgger mode: record episodes with is_human tracking for interventions.",
)
flags.DEFINE_string(
    "entry_prefix",
    "",
    "Entry prefix for saved episodes (required when enable_dagger=True).",
)
flags.DEFINE_string(
    "reset_trajectory",
    "Pre-insert motion Rectify",
    "Trajectory name to use for reset pose (empty to skip).",
)

# Pedal support (used automatically in dagger mode)
flags.DEFINE_string(
    "pedal_device",
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd",
    "Device path for the foot pedal.",
)

# Model preloading
flags.DEFINE_bool(
    "preload_models",
    False,
    "If True, preload the model as an inference service before execution. "
    "This eliminates model load time for faster inference. "
    "Requires --model_id to be set.",
)

flags.DEFINE_integer(
    "preload_timeout",
    120,
    "Timeout in seconds when waiting for model services to be ready",
)


class _DaggerState(enum.StrEnum):
  """State machine for DAgger controller."""

  IDLE = "idle"
  BUSY = "busy"
  TELEOP = "teleop"
  POLICY = "policy"
  ALIGNING = "aligning"


class _Button:
  """Single button with pop-event semantics."""

  def __init__(self) -> None:
    self._pressed = False
    self._lock = threading.Lock()

  def set_pressed(self) -> None:
    with self._lock:
      self._pressed = True

  def pop_event(self) -> bool:
    """Return True if pressed since last call, then reset."""
    with self._lock:
      pressed = self._pressed
      self._pressed = False
      return pressed


class PedalListener:
  """Listen to a 3-button foot pedal via evdev with pop-event semantics."""

  def __init__(self, device_path: str):
    self._device_path = device_path
    self._device = None
    self._thread: threading.Thread | None = None
    self._running = False

    # Buttons with pop-event semantics
    self.a = _Button()
    self.b = _Button()
    self.c = _Button()

  def start(self):
    if self._running:
      return
    self._running = True
    try:
      self._device = InputDevice(self._device_path)
    except FileNotFoundError as e:
      raise ValueError(f"Pedal device not found: {self._device_path}") from e
    self._thread = threading.Thread(target=self._event_loop, daemon=True)
    self._thread.start()

  def stop(self):
    self._running = False
    if self._thread:
      self._thread.join(timeout=1.0)
    if self._device:
      self._device.close()

  def _event_loop(self):
    for event in self._device.read_loop():
      if not self._running:
        break
      if event.type == ecodes.EV_KEY and event.value == 1:  # Key press
        self._handle_button(event.code)

  def _handle_button(self, code: int):
    if code == ecodes.KEY_A:
      self.a.set_pressed()
    elif code == ecodes.KEY_B:
      self.b.set_pressed()
    elif code == ecodes.KEY_C:
      self.c.set_pressed()


class TerminationMonitor:
  """Monitors task progress and determines when to terminate."""

  def __init__(
      self,
      query_client: r2client.QueryClient,
      threshold: float,
      min_frames: int,
      model_id: str = "",
      service_address: str = "",
  ):
    self._query_client = query_client
    self._model_id = model_id
    self._service_address = service_address
    self._threshold = threshold
    self._min_frames = min_frames
    self._frames_above = 0

  def check(self) -> bool:
    """Check progress and return True if termination threshold reached."""
    response = self._query_client.predict_progress(
        model_id=self._model_id,
        service_address=self._service_address,
    )

    # Handle error or missing progress (fail safe - don't terminate on error)
    if response.error or response.progress is None:
      if response.error:
        log.warning("Termination check failed: {}", response.error)
      else:
        log.warning("Termination check returned no progress value")
      return False

    progress = response.progress
    done = progress >= self._threshold
    log.debug("Progress: {:.3f}, done: {}", progress, done)

    if done:
      self._frames_above += 1
    else:
      self._frames_above = 0

    if self._frames_above >= self._min_frames:
      log.info(
          "Progress {:.3f} >= {} for {} frames - terminating",
          progress,
          self._threshold,
          self._frames_above,
      )
      return True

    return False

  def reset(self) -> None:
    """Reset frame counter for new episode."""
    self._frames_above = 0


def _monitor_progress(
    motion_future: futures.Future[rpc_api.TicketStatusResponse],
    termination_monitor: TerminationMonitor,
    poll_interval: float,
    stop_event: threading.Event,
) -> None:
  """Monitor progress in background thread."""
  termination_monitor.reset()

  while not motion_future.done() and not stop_event.is_set():
    try:
      if termination_monitor.check():
        stop_event.set()
        break
    except Exception as e:
      log.warning("Termination check failed: {}", e)

    time.sleep(poll_interval)


def _build_query() -> rpc_api.ExecuteLearnedBehaviorQuery:
  """Build the learned behavior query from flags."""
  return rpc_api.ExecuteLearnedBehaviorQuery(
      model_id=FLAGS.model_id,
      service_address=FLAGS.service_address,
      timeout_seconds=FLAGS.timeout,
      obs_history_len=FLAGS.obs_history_len,
      buffer_actions=FLAGS.buffer_actions,
      action_offset=FLAGS.action_offset,
      action_key=FLAGS.action_key,
  )


def _align_leader(robot: r2client.Robot) -> None:
  """Align leader arm with follower for teleop."""
  log.info("Aligning leader arm...")
  try:
    robot.behaviour.align_leader_with_follower(
        timeout_seconds=1.0, threshold=0.1
    ).result()
  except Exception as e:
    log.warning("Align failed: {}", e)


def _run_simple_mode(
    robot: r2client.Robot,
    termination_monitor: TerminationMonitor | None,
) -> None:
  """Run in simple mode without DAgger recording.

  Workflow:
    1. Start in teleop mode
    2. Press Enter to start policy
    3. Press Enter to stop policy and align leader arm
    4. Press Enter again to enable teleop control
    5. Press Enter to resume policy
    6. Ctrl+C to quit
  """
  state = {
      "motion_future": None,
      "running": True,
      "stop_event": None,
  }

  def cancel_motion():
    if state["stop_event"]:
      state["stop_event"].set()
    if state["motion_future"] and not state["motion_future"].done():
      log.warning("Cancelling behavior...")
      state["motion_future"].cancel()

  def cleanup(signum, frame):
    del signum, frame
    state["running"] = False
    cancel_motion()
    raise KeyboardInterrupt

  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  if termination_monitor:
    source = FLAGS.termination_service_address or FLAGS.termination_model_id
    log.info("Termination model: {}", source)
    log.info(
        "Threshold: {}, min frames: {}",
        FLAGS.termination_threshold,
        FLAGS.termination_min_frames,
    )

  def start_policy():
    """Start the learned behavior policy."""
    robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)
    source = FLAGS.service_address or FLAGS.model_id
    log.opt(colors=True).info(
        f"<cyan>Executing: {source}</cyan>",
    )

    state["motion_future"] = robot.behaviour.execute_learned_behavior(
        _build_query()
    )

  def switch_to_teleop():
    """Switch to teleop mode for human control."""
    cancel_motion()
    _align_leader(robot)
    log.opt(colors=True).info("<yellow>Leader arm aligned.</yellow>")
    input("Press Enter to enable teleop control...")
    robot.exec_mode.set_execution_mode(
        rpc_api.ExecutionMode.DATA_COLLECTION_TELEOP
    )
    log.opt(colors=True).info(
        "<green>Human control active (gello mode)</green>"
    )

  try:
    while state["running"]:
      robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

      if FLAGS.reset_trajectory:
        log.opt(colors=True).info(
            f"<magenta>Moving to reset pose ({FLAGS.reset_trajectory})...</magenta>",
        )
        state["motion_future"] = robot.behaviour.trajectory_motion(
            trajectory_name=FLAGS.reset_trajectory,
            motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
            static_gripper=False,
        )
        try:
          state["motion_future"].result()
        except Exception as e:
          log.warning("Reset trajectory failed: {}", e)

      # Start in teleop mode
      switch_to_teleop()

      # Teleop <-> policy cycle (no reset pose between interventions)
      while state["running"]:
        input("Press Enter to start policy or Ctrl+C to quit...")
        start_policy()

        # Start progress monitoring if termination model is configured
        monitor_thread = None
        state["stop_event"] = threading.Event()
        if termination_monitor:
          log.opt(colors=True).info(
              "<cyan>Monitoring progress..."
              " (Enter to stop/switch to teleop)</cyan>"
          )
          monitor_thread = threading.Thread(
              target=_monitor_progress,
              args=(
                  state["motion_future"],
                  termination_monitor,
                  FLAGS.poll_interval,
                  state["stop_event"],
              ),
              daemon=True,
          )
          monitor_thread.start()
        else:
          log.opt(colors=True).info(
              "<cyan>Running... (Enter to stop/switch to teleop)</cyan>"
          )

        # Wait for user input, policy completion, or termination
        user_interrupted = False
        while not state["stop_event"].is_set():
          if state["motion_future"] and state["motion_future"].done():
            break
          ready, _, _ = select.select([sys.stdin], [], [], 0.1)
          if ready:
            sys.stdin.readline()
            user_interrupted = True
            break

        if monitor_thread:
          state["stop_event"].set()
          monitor_thread.join(timeout=1.0)

        if user_interrupted:
          log.opt(colors=True).info("<yellow>Switching to teleop...</yellow>")
          switch_to_teleop()
          continue

        # Policy completed naturally (termination or timeout)
        cancel_motion()
        log.success("Done.")
        break
  except KeyboardInterrupt:
    log.info("Stopping...")
  finally:
    cancel_motion()


class DaggerController:
  """Controller for DAgger mode with episode recording and intervention.

  All public methods are non-blocking. Async operations (reset trajectory,
  leader alignment) are polled via tick() from the main loop, keeping
  the event loop responsive so pedal events never go stale.
  """

  def __init__(self, robot: r2client.Robot):
    self._robot = robot
    self._lock = threading.Lock()

    # State
    self._motion_future = None
    self._running = True
    self._state = _DaggerState.IDLE
    self._intervention_count = 0
    self._saved_count = 0
    self._observer_started = False

    # Async operation: (future, callback) or None
    self._pending: tuple | None = None

  @property
  def is_recording(self) -> bool:
    with self._lock:
      return self._state != _DaggerState.IDLE

  @property
  def saved_count(self) -> int:
    with self._lock:
      return self._saved_count

  def stop_running(self):
    with self._lock:
      self._running = False

  def is_running(self) -> bool:
    with self._lock:
      return self._running

  def tick(self):
    """Advance pending async operations. Call from the main loop."""
    if self._pending is None:
      return
    future, callback = self._pending
    if not future.done():
      return
    self._pending = None
    try:
      future.result()
    except Exception as e:
      log.warning("{}", e)
    callback()

  def _run_async(self, future, callback):
    """Schedule an async operation with a completion callback."""
    with self._lock:
      self._state = _DaggerState.BUSY
    self._pending = (future, callback)

  def _rec_tag(self) -> str:
    """Return a recording indicator for status messages."""
    with self._lock:
      if self._observer_started and self._state in (
          _DaggerState.POLICY,
          _DaggerState.TELEOP,
      ):
        return " <red>● REC</red>"
    return ""

  def cancel_motion(self):
    self._pending = None
    with self._lock:
      if self._motion_future and not self._motion_future.done():
        self._motion_future.cancel()
      self._motion_future = None

  def start_episode(self):
    """Start a new episode: reset pose -> align -> teleop."""
    if self.is_recording:
      log.warning("Already recording!")
      return

    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    with self._lock:
      self._intervention_count = 0
      self._observer_started = False

    if FLAGS.reset_trajectory:
      log.opt(colors=True).info(
          f"<magenta>Moving to reset pose ({FLAGS.reset_trajectory})...</magenta>",
      )
      future = self._robot.behaviour.trajectory_motion(
          trajectory_name=FLAGS.reset_trajectory,
          motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
          static_gripper=False,
      )
      self._run_async(future, self._start_align)
    else:
      self._start_align()

  def _start_align(self):
    """Start leader alignment (async)."""
    log.info("Sending alignment request...")
    future = self._robot.behaviour.align_leader_with_follower(
        timeout_seconds=1.0, threshold=0.1
    )
    self._run_async(future, self._enter_aligning)

  def _start_policy(self):
    """Start policy execution."""
    try:
      health_status = self._robot.hardware_health.get_status()
      if not health_status.is_healthy:
        log.error(
            "Cannot start policy: hardware unhealthy ({})",
            health_status.summary,
        )
        with self._lock:
          self._state = _DaggerState.TELEOP
        return
    except Exception as error:  # pylint: disable=broad-exception-caught
      log.warning("Failed to fetch hardware health status: {}", error)

    with self._lock:
      self._state = _DaggerState.POLICY
      self._observer_started = True
    self._robot.episode_observer.set_is_human(False)
    self._robot.episode_observer.start()
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    source = FLAGS.service_address or FLAGS.model_id
    log.opt(colors=True).info(
        f"<cyan>{self._rec_tag()} Executing: {source}</cyan>",
    )

    with self._lock:
      self._motion_future = self._robot.behaviour.execute_learned_behavior(
          _build_query(), timeout=FLAGS.timeout
      )
    self._print_actions()

  def switch_to_teleop(self):
    """Cancel policy and start alignment (async)."""
    log.info("Stopping policy...")
    self.cancel_motion()
    if self._observer_started:
      self._robot.episode_observer.stop()
    log.info("Sending alignment request...")
    future = self._robot.behaviour.align_leader_with_follower(
        timeout_seconds=1.0, threshold=0.1
    )
    self._run_async(future, self._enter_aligning)

  def _enter_aligning(self):
    """Alignment done, wait for user to confirm teleop."""
    with self._lock:
      self._state = _DaggerState.ALIGNING
    log.opt(colors=True).info(
        f"<yellow>{self._rec_tag()} Leader arm aligned.</yellow>",
    )
    self._print_actions()

  def confirm_teleop(self):
    """Confirm teleop after alignment - actually enables human control."""
    self._robot.exec_mode.set_execution_mode(
        rpc_api.ExecutionMode.DATA_COLLECTION_TELEOP
    )

    with self._lock:
      self._state = _DaggerState.TELEOP
      is_intervention = self._observer_started
      if is_intervention:
        self._robot.episode_observer.start()
        self._robot.episode_observer.set_is_human(True)
        self._intervention_count += 1
        count = self._intervention_count

    if is_intervention:
      log.opt(colors=True).info(
          f"<green>{self._rec_tag()} Human control active (intervention #{count})</green>",
      )
    else:
      log.opt(colors=True).info(
          "<green>Human control active. Press A to start policy.</green>"
      )
    self._print_actions()

  def resume_policy(self):
    """Resume policy after teleop."""
    log.opt(colors=True).info(
        f"<cyan>{self._rec_tag()} Resuming...</cyan>",
    )
    self._start_policy()

  def toggle_control(self):
    """Toggle between policy and teleop based on current state."""
    with self._lock:
      state = self._state
    match state:
      case _DaggerState.IDLE:
        self.start_episode()
      case _DaggerState.BUSY:
        log.warning("Please wait...")
      case _DaggerState.ALIGNING:
        self.confirm_teleop()
      case _DaggerState.TELEOP:
        self.resume_policy()
      case _DaggerState.POLICY:
        self.switch_to_teleop()

  def _print_actions(self):
    """Print available pedal actions for the current state."""
    with self._lock:
      state = self._state
    c = log.opt(colors=True)
    if state == _DaggerState.IDLE:
      c.info("<bold><cyan>A</cyan></bold>: Start new episode")
    elif state == _DaggerState.TELEOP:
      c.info(
          "<bold><cyan>A</cyan></bold>: Start/resume policy  |  "
          "<bold><green>B</green></bold>: Save  |  "
          "<bold><yellow>C</yellow></bold>: Discard"
      )
    elif state == _DaggerState.POLICY:
      c.info(
          "<bold><cyan>A</cyan></bold>: Stop & switch to teleop  |  "
          "<bold><green>B</green></bold>: Save  |  "
          "<bold><yellow>C</yellow></bold>: Discard"
      )
    elif state == _DaggerState.ALIGNING:
      c.info(
          "<bold><cyan>A</cyan></bold>: Enable teleop  |  "
          "<bold><green>B</green></bold>: Save  |  "
          "<bold><yellow>C</yellow></bold>: Discard"
      )
    elif state == _DaggerState.BUSY:
      log.warning("Please wait...")

  def stop_and_save(self):
    """Stop and save the current episode."""
    if not self.is_recording:
      log.warning("Not recording!")
      return

    self.cancel_motion()

    with self._lock:
      observer_started = self._observer_started
      count = self._intervention_count

    if not observer_started:
      log.warning("No data recorded (policy never ran).")
      with self._lock:
        self._state = _DaggerState.IDLE
      self._print_actions()
      return

    self._robot.episode_observer.stop()
    # NOTE: Ideally we'd switch to TEACH here so the leader arm is freely
    # movable after an episode. However, set_execution_mode controls the
    # *follower* arm's controller. TEACH mode would also put the follower into
    # gravity-comp. Freeing only the leader requires a dedicated RPC endpoint
    # for the gello teach_mode_enable service.
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    desc = f"DAgger [interventions={count}]"
    try:
      self._robot.episode_observer.set_task_description(desc)
    except Exception:
      pass

    self._robot.episode_observer.save(entry_prefix=FLAGS.entry_prefix)

    with self._lock:
      self._state = _DaggerState.IDLE
      self._saved_count += 1
      self._observer_started = False
      saved = self._saved_count

    log.success(
        "Episode saved (interventions={}). Total saved: {}", count, saved
    )
    self._print_actions()

  def stop_and_discard(self):
    """Stop and save the current episode marked as discarded."""
    if not self.is_recording:
      log.warning("Not recording!")
      return

    self.cancel_motion()

    with self._lock:
      observer_started = self._observer_started
      count = self._intervention_count

    if not observer_started:
      log.warning("No data recorded (policy never ran).")
      with self._lock:
        self._state = _DaggerState.IDLE
      self._print_actions()
      return

    self._robot.episode_observer.stop()
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    desc = f"DAgger DISCARDED [interventions={count}]"
    try:
      self._robot.episode_observer.set_task_description(desc)
    except Exception:
      pass

    self._robot.episode_observer.save(
        entry_prefix=f"discarded_{FLAGS.entry_prefix}"
    )

    with self._lock:
      self._state = _DaggerState.IDLE
      self._observer_started = False

    log.warning("Episode saved as discarded.")
    self._print_actions()

  def emergency_stop(self):
    """Emergency stop - save episode as discarded."""
    log.error("Emergency stop!")
    self.cancel_motion()
    if self.is_recording:
      with self._lock:
        observer_started = self._observer_started
      if observer_started:
        try:
          self._robot.episode_observer.stop()
          self._robot.episode_observer.discard()
        except Exception:
          pass
      with self._lock:
        self._state = _DaggerState.IDLE
        self._observer_started = False
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  def cleanup(self):
    """Final cleanup."""
    self.cancel_motion()
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)


def _run_dagger_mode(robot: r2client.Robot) -> None:
  """Run in DAgger mode with episode recording and intervention support."""
  ctrl = DaggerController(robot)

  pedal = PedalListener(FLAGS.pedal_device)
  pedal.start()
  log.success("Pedal connected.")

  def cleanup_handler(signum, frame):
    del signum, frame
    ctrl.stop_running()
    ctrl.emergency_stop()
    raise KeyboardInterrupt

  signal.signal(signal.SIGINT, cleanup_handler)
  signal.signal(signal.SIGTERM, cleanup_handler)

  source = FLAGS.service_address or FLAGS.model_id
  log.opt(colors=True).info(
      f"DAgger mode. Model: {source}\n"
      "  Pedal controls:\n"
      "    <bold><cyan>A</cyan></bold> (start/resume): Start episode or resume policy\n"
      "    <bold><cyan>A</cyan></bold> (during policy): Stop and align leader arm\n"
      "    <bold><cyan>A</cyan></bold> (after alignment): Enable teleop control\n"
      "    <bold><green>B</green></bold>: Save current episode\n"
      "    <bold><yellow>C</yellow></bold>: Discard current episode\n"
      "  Ctrl+C to quit."
  )

  try:
    while ctrl.is_running():
      ctrl.tick()
      if pedal.a.pop_event():
        ctrl.toggle_control()
      elif pedal.b.pop_event():
        if ctrl.is_recording:
          ctrl.stop_and_save()
      elif pedal.c.pop_event():
        if ctrl.is_recording:
          ctrl.stop_and_discard()
      time.sleep(0.05)
  except KeyboardInterrupt:
    log.info("Stopping...")
  finally:
    pedal.stop()
    ctrl.cleanup()
    log.success("Done. Saved {} episodes.", ctrl.saved_count)


def main(_):
  dotenv.load_dotenv()
  r2_logging.configure(service="learned-behavior")
  sentry.init_sentry(service="learned-behavior")

  if not FLAGS.model_id and not FLAGS.service_address:
    raise ValueError("Specify --model_id or --service_address")

  if FLAGS.enable_dagger and not FLAGS.entry_prefix:
    raise ValueError("--entry_prefix is required when --enable_dagger is set")

  robot = r2client.Robot(
      server_address=f"tcp://{FLAGS.server}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.server}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://localhost:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  # Create termination monitor if configured
  termination_monitor = None
  if FLAGS.termination_model_id or FLAGS.termination_service_address:
    if FLAGS.termination_model_id and FLAGS.termination_service_address:
      raise ValueError(
          "Specify either --termination_model_id or --termination_service_address, "
          "not both"
      )
    termination_monitor = TerminationMonitor(
        query_client=robot.query,
        threshold=FLAGS.termination_threshold,
        min_frames=FLAGS.termination_min_frames,
        model_id=FLAGS.termination_model_id,
        service_address=FLAGS.termination_service_address,
    )

  # Preload model if requested
  if FLAGS.preload_models:
    if not FLAGS.model_id:
      raise ValueError("--preload_models requires --model_id to be set")

    # Check whether the model service is already running.
    existing_services = robot.model_services.get_all()
    model_running = False
    for svc in existing_services:
      if svc.model_id == FLAGS.model_id and svc.healthy:
        log.info("Model service already running at: {}", svc.address)
        model_running = True
        break

    if not model_running:
      log.info("Preloading model: {}", FLAGS.model_id)
      address = robot.model_services.start(FLAGS.model_id)

      log.info("Waiting for model to be ready...")
      response = robot.model_services.wait_until_ready(
          model_ids=[FLAGS.model_id],
          timeout=FLAGS.preload_timeout,
      )

      if not response.success:
        raise RuntimeError(
            f"Model service failed to start. Pending: {response.pending_models}"
        )

      log.success("Model service ready at: {}", address)
    log.info("Learned behavior will automatically use the preloaded service")

  if FLAGS.enable_dagger:
    _run_dagger_mode(robot)
  else:
    _run_simple_mode(robot, termination_monitor)


if __name__ == "__main__":
  try:
    app.run(main)
  except SystemExit:
    raise
  except KeyboardInterrupt:
    pass
  except Exception:
    sentry.capture_exception()
    raise
