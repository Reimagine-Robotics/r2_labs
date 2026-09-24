from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "name", None, "The name of the trajectory to execute", required=True
)

flags.DEFINE_float(
    "speed", 1.0, "Speed multiple for a full replay; 1.0 replays as recorded"
)

flags.DEFINE_float(
    "go_to_duration",
    -1.0,
    "Seconds a go-to move takes; <= 0 uses the default timeout",
)

flags.DEFINE_bool("static_gripper", False, "Whether to keep the gripper static")

flags.DEFINE_bool(
    "steady_pacing",
    False,
    "Replay a full joint motion at the robot's steady traverse rate rather than"
    " the taught pace; grasps and presses keep their taught timing.",
)

flags.DEFINE_float(
    "allowance_factor",
    1.0,
    "How much more coarsely than the recording steady pacing may cut a corner;"
    " 1.0 keeps to the recorded detail. Ignored without --steady_pacing.",
)

flags.DEFINE_enum(
    "motion_type", "full", ["full", "start", "end"], "Motion type"
)

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
      speed=FLAGS.speed,
      steady_pacing=FLAGS.steady_pacing,
      allowance_factor=FLAGS.allowance_factor,
      go_to_duration=(
          None if FLAGS.go_to_duration <= 0.0 else FLAGS.go_to_duration
      ),
  )

  print("Moving ...")
  motion_future.result()


if __name__ == "__main__":
  app.run(main)
