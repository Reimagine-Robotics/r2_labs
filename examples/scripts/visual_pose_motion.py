from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "name", None, "The name of the visual pose to move to", required=True
)

flags.DEFINE_float("period", 5.0, "Period of the trajectory")


def main(_):

  robot = r2client.Robot(
      server_address=f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://localhost:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  motion_future = robot.behaviour.visual_pose_motion(
      visual_pose_name=FLAGS.name,
      period_seconds=FLAGS.period,
  )

  print("Moving ...")
  motion_future.result()


if __name__ == "__main__":
  app.run(main)
