"""CLI for executing learned behaviors via local or remote inference.

Simple usage (default):
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160"

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

Pedal controls (DAgger mode):
  - A (first press): Start episode with policy running
  - A (during policy): Align leader arm, wait for confirmation
  - A (after alignment): Confirm and enable teleop control
  - A (during teleop): Resume policy execution
  - B: Save current episode
  - C: Discard current episode
  - Ctrl+C: Quit
"""

from __future__ import annotations

import enum
import select
import signal
import sys
import threading
import time

import dotenv
from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api
from r2_labs.sdk import futures
from r2_labs.sdk import logging as r2_logging
from r2_labs.sdk import sentry

import evdev
from evdev import InputDevice, ecodes

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


class PedalButton(enum.Enum):
  """Pedal button identifiers."""

  A = 0
  B = 1
  C = 2
  OTHER = 3


class PedalListener:
  """Listen to a 3-button foot pedal via evdev."""

  def __init__(self, device_path: str, on_press):
    self._device_path = device_path
    self._device = None
    self._thread: threading.Thread | None = None
    self._running = False
    self._on_press = on_press

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
        button = self._code_to_button(event.code)
        if button != PedalButton.OTHER:
          self._on_press(button)

  def _code_to_button(self, code: int) -> PedalButton:
    if code == ecodes.KEY_A:
      return PedalButton.A
    if code == ecodes.KEY_B:
      return PedalButton.B
    if code == ecodes.KEY_C:
      return PedalButton.C
    return PedalButton.OTHER


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
        print(f"Warning: termination check failed: {response.error}")
      else:
        print("Warning: termination check returned no progress value")
      return False

    progress = response.progress
    done = progress >= self._threshold
    print(f"Progress: {progress:.3f}, done: {done}")

    if done:
      self._frames_above += 1
    else:
      self._frames_above = 0

    if self._frames_above >= self._min_frames:
      print(
          f"Progress {progress:.3f} >= {self._threshold} "
          f"for {self._frames_above} frames - terminating"
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
      print(f"Warning: termination check failed: {e}")

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


def _run_simple_mode(
    robot: r2client.Robot,
    termination_monitor: TerminationMonitor | None,
) -> None:
  """Run in simple mode without DAgger recording."""
  state = {"motion_future": None, "running": True, "stop_event": None}

  def cancel_motion():
    if state["stop_event"]:
      state["stop_event"].set()
    if state["motion_future"] and not state["motion_future"].done():
      print("\nCancelling behavior...")
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
    print(f"Termination model: {source}")
    print(
        f"Threshold: {FLAGS.termination_threshold}, "
        f"min frames: {FLAGS.termination_min_frames}"
    )

  try:
    while state["running"]:
      robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

      if FLAGS.reset_trajectory:
        print(f"Moving to reset pose ({FLAGS.reset_trajectory})...")
        state["motion_future"] = robot.behaviour.trajectory_motion(
            trajectory_name=FLAGS.reset_trajectory,
            motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
            static_gripper=False,
        )
        state["motion_future"].result()

      input("Press Enter to start learned behavior...")

      robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)
      source = FLAGS.service_address or FLAGS.model_id
      print(f"Executing learned behavior from: {source}")

      state["motion_future"] = robot.behaviour.execute_learned_behavior(
          _build_query()
      )

      # Start progress monitoring if termination model is configured
      monitor_thread = None
      if termination_monitor:
        print("Behavior executing... (monitoring progress, Enter to stop)")
        state["stop_event"] = threading.Event()
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

        # Wait for either termination or user input
        while not state["stop_event"].is_set():
          # Check for user input with timeout
          ready, _, _ = select.select([sys.stdin], [], [], 0.1)
          if ready:
            sys.stdin.readline()  # Consume the input
            break

        state["stop_event"].set()
        monitor_thread.join(timeout=1.0)
      else:
        input("Press Enter to stop...")

      cancel_motion()
      print("Done.\n")
  except KeyboardInterrupt:
    print("\nStopping...")
  finally:
    cancel_motion()


class DaggerController:
  """Controller for DAgger mode with episode recording and intervention."""

  def __init__(self, robot: r2client.Robot):
    self._robot = robot
    self._lock = threading.Lock()

    # State
    self._motion_future = None
    self._running = True
    self._is_human = False
    self._recording = False
    self._intervention_count = 0
    self._saved_count = 0
    self._waiting_for_teleop_confirm = False

  @property
  def is_recording(self) -> bool:
    with self._lock:
      return self._recording

  @property
  def is_human(self) -> bool:
    with self._lock:
      return self._is_human

  @property
  def saved_count(self) -> int:
    with self._lock:
      return self._saved_count

  @property
  def waiting_for_teleop_confirm(self) -> bool:
    with self._lock:
      return self._waiting_for_teleop_confirm

  def stop_running(self):
    with self._lock:
      self._running = False

  def is_running(self) -> bool:
    with self._lock:
      return self._running

  def cancel_motion(self):
    with self._lock:
      if self._motion_future and not self._motion_future.done():
        self._motion_future.cancel()
      self._motion_future = None

  def move_to_reset(self):
    """Move to reset pose if configured."""
    if not FLAGS.reset_trajectory:
      return
    print(f"Moving to reset pose ({FLAGS.reset_trajectory})...")
    future = self._robot.behaviour.trajectory_motion(
        trajectory_name=FLAGS.reset_trajectory,
        motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
        static_gripper=False,
    )
    future.result()

  def align_leader(self):
    """Align leader arm with follower for teleop."""
    print("Aligning leader arm...")
    try:
      self._robot.behaviour.align_leader_with_follower(
          timeout_seconds=3.0, threshold=0.1
      ).result()
    except Exception as e:
      print(f"Warning: align failed: {e}")

  def start_episode(self):
    """Start a new episode."""
    if self.is_recording:
      print("Already recording!")
      return

    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)
    self.move_to_reset()
    self.align_leader()

    with self._lock:
      self._intervention_count = 0
      self._is_human = False
      self._recording = True

    self._robot.episode_observer.set_is_human(False)
    self._robot.episode_observer.start()

    self._start_policy()

  def _start_policy(self):
    """Start policy execution."""
    with self._lock:
      self._is_human = False
    self._robot.episode_observer.set_is_human(False)
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    source = FLAGS.service_address or FLAGS.model_id
    print(f"[POLICY] Executing: {source}")

    with self._lock:
      self._motion_future = self._robot.behaviour.execute_learned_behavior(
          _build_query(), timeout=FLAGS.timeout
      )

  def switch_to_teleop(self):
    """Start switching to human teleop - aligns leader and waits for confirm."""
    if not self.is_recording:
      print("Not recording!")
      return

    self.cancel_motion()
    self.align_leader()

    with self._lock:
      self._waiting_for_teleop_confirm = True

    print("[ALIGNED] Press pedal A again to enable teleop...")

  def confirm_teleop(self):
    """Confirm teleop after alignment - actually enables human control."""
    self._robot.exec_mode.set_execution_mode(
        rpc_api.ExecutionMode.DATA_COLLECTION_TELEOP
    )
    self._robot.episode_observer.set_is_human(True)

    with self._lock:
      self._is_human = True
      self._waiting_for_teleop_confirm = False
      self._intervention_count += 1
      count = self._intervention_count

    print(f"[TELEOP] Human control active (intervention #{count})")

  def resume_policy(self):
    """Resume policy after teleop."""
    if not self.is_recording:
      print("Not recording!")
      return
    print("[POLICY] Resuming...")
    self._start_policy()

  def toggle_control(self):
    """Toggle between policy and teleop."""
    if not self.is_recording:
      self.start_episode()
    elif self.waiting_for_teleop_confirm:
      self.confirm_teleop()
    elif self.is_human:
      self.resume_policy()
    else:
      self.switch_to_teleop()

  def stop_and_save(self):
    """Stop and save the current episode."""
    if not self.is_recording:
      print("Not recording!")
      return

    self.cancel_motion()
    self._robot.episode_observer.stop()
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    with self._lock:
      count = self._intervention_count

    desc = f"DAgger [interventions={count}]"
    try:
      self._robot.episode_observer.set_task_description(desc)
    except Exception:
      pass

    self._robot.episode_observer.save(entry_prefix=FLAGS.entry_prefix)

    with self._lock:
      self._recording = False
      self._saved_count += 1
      saved = self._saved_count

    print(f"Episode saved (interventions={count}). Total saved: {saved}")

  def stop_and_discard(self):
    """Stop and discard the current episode."""
    if not self.is_recording:
      print("Not recording!")
      return

    self.cancel_motion()
    self._robot.episode_observer.stop()
    self._robot.episode_observer.discard()
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

    with self._lock:
      self._recording = False

    print("Episode discarded.")

  def emergency_stop(self):
    """Emergency stop - cancel and discard."""
    print("Emergency stop!")
    self.cancel_motion()
    if self.is_recording:
      try:
        self._robot.episode_observer.stop()
        self._robot.episode_observer.discard()
      except Exception:
        pass
      with self._lock:
        self._recording = False
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  def cleanup(self):
    """Final cleanup."""
    self.cancel_motion()
    self._robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)


def _run_dagger_mode(robot: r2client.Robot) -> None:
  """Run in DAgger mode with episode recording and intervention support."""
  ctrl = DaggerController(robot)

  def cleanup_handler(signum, frame):
    del signum, frame
    ctrl.stop_running()
    ctrl.emergency_stop()
    raise KeyboardInterrupt

  signal.signal(signal.SIGINT, cleanup_handler)
  signal.signal(signal.SIGTERM, cleanup_handler)

  # Setup pedal
  def on_pedal(button: PedalButton):
    if button == PedalButton.A:
      ctrl.toggle_control()
    elif button == PedalButton.B:
      if ctrl.is_recording:
        ctrl.stop_and_save()
    elif button == PedalButton.C:
      if ctrl.is_recording:
        ctrl.stop_and_discard()

  pedal = PedalListener(FLAGS.pedal_device, on_pedal)
  pedal.start()
  print("Pedal connected.")

  # Print instructions
  source = FLAGS.service_address or FLAGS.model_id
  print(f"\nDAgger mode. Model: {source}")
  print("Pedal controls:")
  print("  A (first press): Start episode with policy running")
  print("  A (during policy): Align leader arm, wait for confirmation")
  print("  A (after alignment): Confirm and enable teleop control")
  print("  A (during teleop): Resume policy execution")
  print("  B: Save current episode")
  print("  C: Discard current episode")
  print("Ctrl+C to quit.\n")

  try:
    while ctrl.is_running():
      signal.pause()
  except KeyboardInterrupt:
    print("\nStopping...")
  finally:
    pedal.stop()
    ctrl.cleanup()
    print(f"Done. Saved {ctrl.saved_count} episodes.")


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
        print(f"Model service already running at: {svc.address}")
        model_running = True
        break

    if not model_running:
      print(f"Preloading model: {FLAGS.model_id}")
      address = robot.model_services.start(FLAGS.model_id)

      print("Waiting for model to be ready...")
      response = robot.model_services.wait_until_ready(
          model_ids=[FLAGS.model_id],
          timeout=FLAGS.preload_timeout,
      )

      if not response.success:
        raise RuntimeError(
            f"Model service failed to start. Pending: {response.pending_models}"
        )

      print(f"✓ Model service ready at: {address}")
    print("  Learned behavior will automatically use the preloaded service\n")

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
