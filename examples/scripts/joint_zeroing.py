"""Calibrates every arm joint against its hard stop.

The arm moves through each joint's range from wherever it is, so clear the
workspace first. The calibration lasts until the arm's next enable.

uv run python r2_labs/examples/scripts/joint_zeroing.py --hostname <robot>
"""

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "hostname",
    "localhost",
    "Hostname of the robot running the RPC API service.",
)


def main(_):
  robot = r2client.Robot(
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  cur_mode = robot.exec_mode.get_execution_mode()
  robot.exec_mode.set_execution_mode(new_mode=rpc_api.ExecutionMode.READY)

  print("Zeroing joints ...")
  info = robot.behaviour.joint_zeroing().result().info
  assert info is not None
  print(f"{info.termination_reason}: {info.result_data or info.error_message}")

  robot.exec_mode.set_execution_mode(new_mode=cur_mode.current_mode)


if __name__ == "__main__":
  app.run(main)
