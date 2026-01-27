"""CLI for executing learned behaviors via local or remote inference.

Simple usage (default):
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160"

DAgger mode (with pedal + episode recording):
  uv run python r2_labs/examples/scripts/learned_behavior.py \
    --server=localhost \
    --model_id="DCAM#tender-engineer-160" \
    --enable_dagger \
    --entry_prefix=dagger_dcam

Pedal controls (DAgger mode):
  - A (first press): Start episode with policy running
  - A (during policy): Align leader arm, wait for confirmation
  - A (after alignment): Confirm and enable teleop control
  - A (during teleop): Resume policy execution
  - B: Discard current episode
  - C: Save current episode
  - Ctrl+C: Quit
"""

from __future__ import annotations

import enum
import signal
import threading

import dotenv
from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

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


def _run_simple_mode(robot: r2client.Robot) -> None:
  """Run in simple mode without DAgger recording."""
  state = {"motion_future": None, "running": True}

  def cancel_motion():
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
        ctrl.stop_and_discard()
    elif button == PedalButton.C:
      if ctrl.is_recording:
        ctrl.stop_and_save()

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
  print("  B: Discard current episode")
  print("  C: Save current episode")
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

  if not FLAGS.model_id and not FLAGS.service_address:
    raise ValueError("Specify --model_id or --service_address")

  if FLAGS.enable_dagger and not FLAGS.entry_prefix:
    raise ValueError("--entry_prefix is required when --enable_dagger is set")

  robot = r2client.Robot(
      server_address=f"tcp://{FLAGS.server}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.server}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://localhost:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  if FLAGS.enable_dagger:
    _run_dagger_mode(robot)
  else:
    _run_simple_mode(robot)


if __name__ == "__main__":
  app.run(main)
