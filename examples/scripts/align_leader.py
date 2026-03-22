"""CLI for aligning the leader arm with the follower arm."""

import signal

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_float(
    "timeout_seconds",
    5.0,
    "Maximum seconds for alignment to complete",
)

flags.DEFINE_float(
    "threshold",
    0.1,
    "Joint position threshold for alignment completion",
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

  print("Aligning leader arm with follower...")

  motion_future = robot.behaviour.align_leader_with_follower(
      timeout_seconds=FLAGS.timeout_seconds,
      threshold=FLAGS.threshold,
  )

  def cleanup(signum, frame):
    del signum, frame  # Unused.
    print("\nCancelling alignment...")
    motion_future.cancel()

  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  motion_future.result()
  print("Done")


if __name__ == "__main__":
  app.run(main)
