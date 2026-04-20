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
  calibration_future = robot.behaviour.calibrate_j0()

  print("Calibrating ...")
  calibration_future.result()

  robot.exec_mode.set_execution_mode(new_mode=cur_mode.current_mode)


if __name__ == "__main__":
  app.run(main)
