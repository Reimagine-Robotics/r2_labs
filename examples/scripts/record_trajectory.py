"""CLI for recording robot trajectories and saving them to the library."""

import signal
import sys
import time

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "name",
    None,
    "Name for the recorded trajectory (required for saving)",
)

flags.DEFINE_string(
    "description",
    "",
    "Description of the trajectory",
)

flags.DEFINE_enum(
    "type",
    "joint_absolute",
    ["joint_absolute", "joint_relative", "wrist_cartesian_relative"],
    "Trajectory type: joint_absolute (raw angles), joint_relative (delta from "
    "start), or wrist_cartesian_relative (6-dof wrist poses)",
)

flags.DEFINE_enum(
    "source",
    "robot",
    ["robot", "teleop"],
    "Trajectory source: robot (kinesthetic teaching) or teleop (teleoperation)",
)

flags.DEFINE_float(
    "timeout",
    30.0,
    "Recording timeout in seconds (0 for no timeout)",
)

flags.DEFINE_bool(
    "save",
    True,
    "Save the trajectory to the library after recording",
)

flags.DEFINE_bool(
    "overwrite",
    False,
    "Overwrite existing trajectory with same name",
)

flags.DEFINE_bool(
    "hold",
    False,
    "Hold robot in place until recording starts. If False (default), robot "
    "switches to TEACH/TELEOP mode immediately so you can position it before "
    "recording. If True, mode change is deferred until start.",
)

flags.DEFINE_string(
    "hostname",
    "localhost",
    "Hostname of the robot running the RPC API service.",
)


def get_trajectory_type() -> rpc_api.TrajectoryType:
  """Convert flag value to TrajectoryType enum."""
  match FLAGS.type:
    case "joint_absolute":
      return rpc_api.TrajectoryType.JOINT_ABSOLUTE
    case "joint_relative":
      return rpc_api.TrajectoryType.JOINT_RELATIVE
    case "wrist_cartesian_relative":
      return rpc_api.TrajectoryType.WRIST_CARTESIAN_RELATIVE
    case _:
      raise ValueError(f"Unknown trajectory type: {FLAGS.type}")


def get_trajectory_source() -> rpc_api.TrajectorySource:
  """Convert flag value to TrajectorySource enum."""
  match FLAGS.source:
    case "robot":
      return rpc_api.TrajectorySource.ROBOT
    case "teleop":
      return rpc_api.TrajectorySource.TELEOP
    case _:
      raise ValueError(f"Unknown trajectory source: {FLAGS.source}")


def main(_):

  if FLAGS.save and not FLAGS.name:
    print("Error: --name is required when --save is enabled")
    sys.exit(1)

  robot = r2client.Robot(
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  trajectory_type = get_trajectory_type()
  trajectory_source = get_trajectory_source()
  timeout = FLAGS.timeout if FLAGS.timeout > 0 else None

  print(f"Preparing to record trajectory...")
  print(f"  Type: {FLAGS.type}")
  print(f"  Source: {FLAGS.source}")
  print(f"  Timeout: {timeout}s" if timeout else "  Timeout: None")
  print(f"  Hold until start: {FLAGS.hold}")

  prepare_response = robot.recording.prepare(
      trajectory_type=trajectory_type,
      trajectory_source=trajectory_source,
      timeout_seconds=timeout,
      hold_until_start=FLAGS.hold,
  )

  if prepare_response.error:
    print(f"Error preparing recording: {prepare_response.error}")
    sys.exit(1)

  if FLAGS.hold:
    print("\nRobot held in place. Press Enter to start recording...")
  else:
    print("\nRobot ready to move. Press Enter to start recording...")

  recording_stopped = False

  def handle_interrupt(signum, frame):
    nonlocal recording_stopped
    del signum, frame  # Unused.
    if not recording_stopped:
      print("\nStopping recording...")
      recording_stopped = True

  signal.signal(signal.SIGINT, handle_interrupt)
  signal.signal(signal.SIGTERM, handle_interrupt)

  try:
    input()
  except EOFError:
    pass

  if recording_stopped:
    sys.exit(0)

  start_response = robot.recording.start()
  if start_response.error:
    print(f"Error starting recording: {start_response.error}")
    sys.exit(1)

  print(
      "Recording... Press Enter or Ctrl+C to stop (or use recording-toggle control)"
  )

  start_time = time.time()
  try:
    while not recording_stopped:
      state = robot.recording.get_state()
      if not state.is_recording:
        if state.timed_out:
          print("\nRecording timed out")
        break

      elapsed = time.time() - start_time
      print(
          f"\r  Samples: {state.sample_count}  Elapsed: {elapsed:.1f}s",
          end="",
          flush=True,
      )
      time.sleep(0.2)
  except EOFError:
    pass

  print("\n\nStopping recording...")
  stop_response = robot.recording.stop()

  if stop_response.error:
    print(f"Error stopping recording: {stop_response.error}")
    sys.exit(1)

  trajectory = stop_response.trajectory
  if trajectory is None:
    print("No trajectory recorded")
    sys.exit(1)

  print(f"\nRecorded trajectory:")
  print(f"  Duration: {trajectory.period_seconds:.2f}s")
  print(f"  Samples: {trajectory.trajectory_data.shape[0]}")
  print(f"  Type: {trajectory.trajectory_type.name}")
  print(f"  Source: {trajectory.trajectory_source.name}")

  if FLAGS.save:
    trajectory.name = FLAGS.name
    trajectory.description = FLAGS.description

    print(f"\nSaving trajectory as '{FLAGS.name}'...")
    add_response = robot.trajectory_library.add_entry(
        trajectory=trajectory,
        allow_overwrite=FLAGS.overwrite,
    )

    if add_response.success:
      print("Trajectory saved successfully")
    else:
      print("Failed to save trajectory (name may already exist)")
      sys.exit(1)
  else:
    print("\nTrajectory not saved (--save=false)")

  print("Done")


if __name__ == "__main__":
  app.run(main)
