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

flags.DEFINE_string(
    "hostname",
    "localhost",
    "Hostname of the robot running the RPC API service.",
)

flags.DEFINE_integer(
    "hold_seconds",
    5,
    "Seconds to hold at the target, reporting measured joints once per second.",
)


def main(_):

  np.set_printoptions(
      precision=4, floatmode="fixed", suppress=True, linewidth=100
  )

  robot = r2client.Robot(
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  target = np.array(
      [float(x) for x in FLAGS.joints.split(",")], dtype=np.float32
  )
  print("Moving ...")
  robot.behaviour.go_to_joints(configuration=target).result()

  print(f"Target joints:   {target}")
  for _ in range(FLAGS.hold_seconds):
    measured = robot.raw_robot.get_proprio_data().joint_positions
    assert measured is not None
    report = f"Measured joints: {measured}"
    if measured.shape == target.shape:
      max_err = float(np.max(np.abs(measured - target)))
      report += f"  max err: {max_err:.4f} rad ({np.degrees(max_err):.2f} deg)"
    print(report)
    time.sleep(1.0)

  print("Stopping")
  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.STOP)


if __name__ == "__main__":
  app.run(main)
