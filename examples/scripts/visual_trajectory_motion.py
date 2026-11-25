"""Replays a visual trajectory, optionally stopping it if the arm pushes too hard.

Prints the run's data warehouse entry id, however the run ends, and the peak
force it applied when it ends by itself; Ctrl-C cancels it. Give the ids of
runs you judge good or bad to applied_force_limits.py to choose a limit.
"""

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "name", None, "The name of the visual trajectory to replay", required=True
)

flags.DEFINE_float(
    "max_applied_force",
    None,
    "Force in newtons above which the replay stops. Unlimited when left out.",
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

  response = robot.behaviour.initiate_visual_trajectory_motion(
      visual_trajectory_name=FLAGS.name,
      max_applied_force=FLAGS.max_applied_force,
  )
  if not response.ticket_id:
    raise RuntimeError(f"The replay failed to start: {response.error}")
  ticket_id = response.ticket_id
  print("Moving ...")

  try:
    robot.behaviour.wait_for_ticket(ticket_id)
    outcome = "done"
  except r2client.AppliedForceLimitError as error:
    outcome = f"stopped at the limit of {error.max_applied_force:.1f} N"
  except r2client.BehaviourFailedError as error:
    outcome = f"failed: {error.error_message or error.termination_reason}"
  except KeyboardInterrupt:
    robot.behaviour.cancel_behaviour(ticket_id)
    outcome = "cancelled"

  ticket = robot.behaviour.get_ticket_status(ticket_id).info
  assert ticket is not None
  peak = (ticket.result_data or {}).get("peak_applied_force")
  force = "no peak force reported" if peak is None else f"peak {peak:.1f} N"
  if ticket.entry_id is None:
    print(f"Unrecorded run {outcome}, {force}.")
    return
  # The warehouse links and applied_force_limits.py take the part after the #.
  entry_id = ticket.entry_id.split("#")[-1]
  print(f"Run {entry_id} {outcome}, {force}.")


if __name__ == "__main__":
  app.run(main)
