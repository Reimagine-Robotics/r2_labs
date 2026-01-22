"""CLI for executing learned behaviors via local or remote inference."""

import signal

import dotenv
from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

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
    "Timeout in seconds",
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


def main(_):
  dotenv.load_dotenv()

  if not FLAGS.model_id and not FLAGS.service_address:
    raise ValueError("Specify --model_id or --service_address")

  robot = r2client.Robot(
      server_address=f"tcp://{FLAGS.server}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.server}:{rpc_api.DEFAULT_QUERY_PORT}",
  )

  state = {"motion_future": None, "running": True}

  def cancel_current_motion() -> None:
    motion_future = state["motion_future"]
    if motion_future is not None and not motion_future.done():
      print("\nCancelling behavior...")
      motion_future.cancel()

  def cleanup(signum, frame):
    del signum, frame  # Unused.
    state["running"] = False
    cancel_current_motion()
    raise KeyboardInterrupt

  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  try:
    while state["running"]:
      # First move to a ready state
      robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)
      state["motion_future"] = robot.behaviour.trajectory_motion(
          trajectory_name="Pre-insert motion Rectify",
          motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
          static_gripper=False,
          period_seconds=None,
      )
      print("Moving to reset pose ...")
      state["motion_future"].result()

      # Then wait for user to confirm
      input("Press Enter to continue to learned behavior execution...")

      robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

      query = rpc_api.ExecuteLearnedBehaviorQuery(
          model_id=FLAGS.model_id,
          service_address=FLAGS.service_address,
          timeout_seconds=FLAGS.timeout,
          obs_history_len=FLAGS.obs_history_len,
          buffer_actions=FLAGS.buffer_actions,
          action_offset=FLAGS.action_offset,
          action_key=FLAGS.action_key,
      )

      # Remote takes priority if both specified
      source = FLAGS.service_address or FLAGS.model_id
      print(f"Executing learned behavior from: {source}")

      state["motion_future"] = robot.behaviour.execute_learned_behavior(query)

      input("Press Enter to stop learned behavior execution...")

      cancel_current_motion()
      print("Done. Resetting robot...\n")
  except KeyboardInterrupt:
    print("\nStopping...")
  finally:
    cancel_current_motion()


if __name__ == "__main__":
  app.run(main)
