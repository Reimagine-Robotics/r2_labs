from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "name", None, "The name of the trajectory to execute", required=True
)

flags.DEFINE_float("period", -1.0, "Period of the trajectory")

flags.DEFINE_bool("static_gripper", False, "Whether to keep the gripper static")

flags.DEFINE_enum(
    "motion_type", "full", ["full", "start", "end"], "Motion type"
)


def main(_):

  robot = r2client.Robot(
      f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
  )

  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  match FLAGS.motion_type:
    case "full":
      motion_type = rpc_api.TrajectoryMotionType.FULL
    case "start":
      motion_type = rpc_api.TrajectoryMotionType.GO_TO_START
    case "end":
      motion_type = rpc_api.TrajectoryMotionType.GO_TO_END
    case _:
      raise ValueError(f"Unknown motion type: {FLAGS.motion_type}")

  motion_future = robot.behaviour.trajectory_motion(
      trajectory_name=FLAGS.name,
      motion_type=motion_type,
      static_gripper=FLAGS.static_gripper,
      period_seconds=None if FLAGS.period <= 0.0 else FLAGS.period,
  )

  print("Moving ...")
  motion_future.result()


if __name__ == "__main__":
  app.run(main)
