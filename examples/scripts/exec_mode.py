from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "mode",
    "",
    ["", "stop", "ready", "teach", "teleop"],
    "The new exec mode, if empty simply query the current exec mode.",
)


def main(_):

  robot = r2client.Robot(
      f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
  )

  if FLAGS.mode:
    match FLAGS.mode:
      case "stop":
        mode = rpc_api.ExecutionMode.STOP
      case "ready":
        mode = rpc_api.ExecutionMode.READY
      case "teach":
        mode = rpc_api.ExecutionMode.TEACH
      case "teleop":
        mode = rpc_api.ExecutionMode.TELEOP
      case _:
        raise ValueError(f"Unknown mode: {FLAGS.mode}")

    response = robot.exec_mode.set_execution_mode(mode)
  else:
    response = robot.exec_mode.get_execution_mode()

  print(f"Exec Mode: {response.current_mode}")


if __name__ == "__main__":
  app.run(main)
