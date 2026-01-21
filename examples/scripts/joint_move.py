import time

import numpy as np
from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "joints",
    "0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
    "Joint configuration to move the arm to. Can be 6 or 7 dimensional",
)


def main(_):

  robot = r2client.Robot(
      f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
  )

  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  joints = [float(x) for x in FLAGS.joints.split(",")]

  motion_future = robot.behaviour.go_to_joints(
      configuration=np.array(joints, dtype=np.float32),
  )
  print("Moving ...")
  motion_future.result()

  time.sleep(5.0)

  print("Stopping")
  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.STOP)


if __name__ == "__main__":
  app.run(main)
