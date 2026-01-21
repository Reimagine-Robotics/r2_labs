import numpy as np
from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "name",
    "",
    "The trajectory name to output. If empty all trajectories are shown.",
)


def main(_):

  robot = r2client.Robot(
      f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
  )
  all_trajectories = robot.trajectory_library.list_entries()

  np.set_printoptions(
      precision=4, floatmode="fixed", suppress=True, linewidth=100
  )

  found = False
  for traj in all_trajectories.trajectories:

    if not FLAGS.name or FLAGS.name == traj.name:
      found = True

      print(f"\nNAME: {traj.name}")
      print(f"TYPE: {traj.trajectory_type})")
      print(f"LENGTH (seconds): {traj.period_seconds}")

      if traj.description:
        print(f"DESCRIPTION: {traj.description}")

      print("DATA:")
      for i in range(traj.trajectory_data.shape[0]):
        print(traj.trajectory_data[i])

  if not found and FLAGS.name:
    print(f"No such trajectory: {FLAGS.name}")


if __name__ == "__main__":
  app.run(main)
