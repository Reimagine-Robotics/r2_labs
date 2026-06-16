from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_float("period", 5.0, "Period of the trajectory")

flags.DEFINE_string(
    "hostname",
    "bronze",
    "Hostname of the robot running the RPC API service.",
)

flags.DEFINE_string(
    "hostname2",
    "diamond",
    "Hostname of the robot running the RPC API service.",
)

def main(_):

  # Defining a robot
  robot = r2client.Robot(
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

 # Defining a robot2
  robot2 = r2client.Robot(
      f"tcp://{FLAGS.hostname2}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.hostname2}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{FLAGS.hostname2}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  robot2.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  ########### Action block #################
  ## Do the thing
  motion_future = robot.behaviour.visual_trajectory_motion(
      visual_trajectory_name="test_down",
  )

  ## Wait for it to finish
  print("Moving ...")
  motion_future.result()
  ############################

  ########### Action block #################
  ## Do the thing
  motion_future = robot.behaviour.visual_trajectory_motion(
      visual_trajectory_name="open_pick_up",
  )

  ## Wait for it to finish
  print("Closing")
  motion_future.result()
  ############################

  ########### Action block #################
  ## Do the thing
  motion_future = robot.behaviour.trajectory_motion(
      trajectory_name="move left",
  )

  ## Wait for it to finish
  print("Moving left")
  motion_future.result()
  ############################
    

if __name__ == "__main__":
  app.run(main)
