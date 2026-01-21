import time

import numpy as np
from absl import app

from r2_labs import client as r2client
from r2_labs import rpc_api


def main(_):

  np.set_printoptions(
      precision=4, floatmode="fixed", suppress=True, linewidth=100
  )

  robot = r2client.Robot(
      f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
  )

  # Set the arm into Teach Mode
  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.TEACH)

  # Spin forever, querying the current arm state.
  while True:
    proprio = robot.raw_robot.get_proprio_data()

    print(f"\nJOINT POSITIONS: {proprio.joint_positions}")
    print(f"\nJOINT VELOCITIES: {proprio.joint_velocities}")
    print(f"GRIPPER POSITION: {proprio.gripper_positions}")
    print(f"WRIST POSE: {proprio.wrist_pose}")

    time.sleep(1.0)


if __name__ == "__main__":
  app.run(main)
