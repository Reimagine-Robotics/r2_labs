"""Exercises the actuated column through the SDK.

MOVES THE COLUMN, including a full-travel homing run. Make sure people are
clear of it before running this.

Walks the whole client surface in the order the failures are demonstrable:

  1. a move refused for want of a position reference (only if the column
     starts uncalibrated -- there is no way to discard a reference on demand)
  2. a move refused because homing owns the motor
  3. homing, waited on through its future
  4. a move to a target, waited on through its future
  5. a move cut short by stop(), and the failed ticket that leaves behind
  6. cancelling the motion that owns the column, which stops it
  7. cancelling a queued motion, which must leave the travelling one alone
  8. cancelling a motion a newer one took over, likewise

Run with:
  uv run python -m r2_labs.examples.scripts.move_column --hostname <robot>
"""

import time

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "hostname",
    "localhost",
    "Hostname of the robot running the RPC API service.",
)

flags.DEFINE_float(
    "target_mm",
    200.0,
    "Height to move to for the completed-move demonstration.",
)

flags.DEFINE_float(
    "stop_target_mm",
    500.0,
    "Height to aim at for the stop demonstration; far enough that stop()"
    " lands while the column is still travelling.",
)

flags.DEFINE_float(
    "stop_after_seconds",
    2.0,
    "How long to let the column travel before calling stop().",
)

# Long enough for a full-travel homing run plus the settle the firmware does
# at each end.
_CALIBRATION_TIMEOUT_S = 90.0
_MOVE_TIMEOUT_S = 90.0


def _report_state(
    robot: r2client.Robot, label: str
) -> rpc_api.ColumnStateResponse:
  state = robot.column.get_state()
  print(
      f"  [{label}] height={state.height_mm:.1f}mm"
      f" direction={state.direction.name.lower()}"
      f" calibrated={state.calibrated} limits_enabled={state.limits_enabled}"
      f" connected={state.connected}"
  )
  return state


def _show_move_refused_while_uncalibrated(robot: r2client.Robot) -> None:
  """A move without a position reference must fail at the call."""
  print("\n1. Moving before calibration")
  state = robot.column.get_state()
  if state.calibrated:
    print(
        "  skipped: the column is already calibrated, and a position"
        " reference cannot be discarded on demand. Power-cycle the board to"
        " see this case."
    )
    return

  response = robot.column.initiate_go_to(FLAGS.target_mm)
  if response.error:
    print(f"  refused, as it should be: {response.error}")
  else:
    print("  UNEXPECTED: the move was accepted without a position reference")


def _show_move_refused_while_calibrating(robot: r2client.Robot) -> None:
  """Homing owns the motor, so a move issued during it must be refused."""
  print("\n2. Calibrating, and moving while it runs")
  calibration = robot.column.initiate_calibrate()
  if calibration.error:
    print(f"  calibration refused: {calibration.error}")
    return
  print(f"  homing started, ticket {calibration.ticket_id}")

  # The firmware only reports the motor busy once homing is actually under
  # way, so give it a moment before asking.
  time.sleep(1.0)
  response = robot.column.initiate_go_to(FLAGS.target_mm)
  if response.error:
    print(f"  move refused mid-homing, as it should be: {response.error}")
  else:
    print("  UNEXPECTED: a move was accepted while homing held the motor")

  print("  waiting for homing to finish ...")
  robot.column.wait_for_ticket(
      calibration.ticket_id, timeout=_CALIBRATION_TIMEOUT_S
  )
  _report_state(robot, "homed")


def _show_completed_move(robot: r2client.Robot) -> None:
  """The ordinary case: a future that resolves when the column arrives."""
  print(f"\n3. Moving to {FLAGS.target_mm:.1f}mm")
  future = robot.column.go_to(FLAGS.target_mm, timeout=_MOVE_TIMEOUT_S)
  future.result()
  _report_state(robot, "arrived")


def _show_stopped_move(robot: r2client.Robot) -> None:
  """stop() cuts drive mid-travel, and the ticket records that it did."""
  print(f"\n4. Moving to {FLAGS.stop_target_mm:.1f}mm, then stopping short")
  response = robot.column.initiate_go_to(FLAGS.stop_target_mm)
  if response.error:
    print(f"  move refused: {response.error}")
    return

  time.sleep(FLAGS.stop_after_seconds)
  robot.column.stop()
  print("  stop() issued")

  # A stopped move never reached its target, so its ticket fails and waiting
  # on it raises. Callers that stop deliberately have to expect that.
  try:
    robot.column.wait_for_ticket(response.ticket_id, timeout=_MOVE_TIMEOUT_S)
  except r2client.BehaviourFailedError as exc:
    print(f"  ticket failed, as it should after a stop: {exc}")
  else:
    print("  UNEXPECTED: the ticket completed despite stop()")
  _report_state(robot, "stopped")


def _settle_at(robot: r2client.Robot, height_mm: float) -> None:
  """Put the column somewhere known, so the next demo has room to travel."""
  robot.column.go_to(height_mm, timeout=_MOVE_TIMEOUT_S).result()


def _still_travelling(robot: r2client.Robot) -> bool:
  state = robot.column.get_state()
  return state.direction is not rpc_api.ColumnDirection.STOPPED


def _show_cancelled_owner(robot: r2client.Robot) -> None:
  """Cancelling the motion that owns the column stops the column."""
  print("\n5. Cancelling the motion that owns the column")
  _settle_at(robot, FLAGS.target_mm)

  response = robot.column.initiate_go_to(FLAGS.stop_target_mm)
  if response.error:
    print(f"  move refused: {response.error}")
    return
  time.sleep(FLAGS.stop_after_seconds)

  reply = robot.column.cancel_ticket(response.ticket_id)
  print(f"  cancel_ticket -> success={reply.success}")
  try:
    robot.column.wait_for_ticket(response.ticket_id, timeout=_MOVE_TIMEOUT_S)
  except r2client.BehaviourFailedError as exc:
    print(f"  ticket failed, as it should after a cancel: {exc}")
  else:
    print("  UNEXPECTED: the ticket completed despite being cancelled")
  _report_state(robot, "cancelled")


def _show_cancelled_queued_motion(robot: r2client.Robot) -> None:
  """A queued motion has no ticket yet, so cancelling it must stop nothing.

  Column motions share one executor, so the second future waits for the first.
  Cancelling it used to cut motor drive and halt the travelling motion.
  """
  print("\n6. Cancelling a motion queued behind a travelling one")
  _settle_at(robot, FLAGS.target_mm)

  travelling = robot.column.go_to(FLAGS.stop_target_mm, timeout=_MOVE_TIMEOUT_S)
  queued = robot.column.go_to(FLAGS.target_mm, timeout=_MOVE_TIMEOUT_S)
  time.sleep(FLAGS.stop_after_seconds)

  print(f"  cancelling the queued motion -> {queued.cancel()}")
  if _still_travelling(robot):
    print("  the travelling motion is still going, as it should be")
  else:
    print("  UNEXPECTED: cancelling a queued motion stopped the column")

  travelling.result()
  _report_state(robot, "first motion finished")


def _show_cancelled_superseded_motion(robot: r2client.Robot) -> None:
  """A motion a newer one took over no longer owns the column to stop."""
  print("\n7. Cancelling a motion a newer one superseded")
  _settle_at(robot, FLAGS.target_mm)

  superseded = robot.column.initiate_go_to(FLAGS.stop_target_mm)
  owner = robot.column.initiate_go_to(FLAGS.stop_target_mm)
  if superseded.error or owner.error:
    print(f"  a move was refused: {superseded.error or owner.error}")
    return
  time.sleep(FLAGS.stop_after_seconds)

  reply = robot.column.cancel_ticket(superseded.ticket_id)
  print(f"  cancelling the superseded motion -> success={reply.success}")
  if _still_travelling(robot):
    print("  the newer motion is still going, as it should be")
  else:
    print("  UNEXPECTED: cancelling a superseded motion stopped the column")

  robot.column.wait_for_ticket(owner.ticket_id, timeout=_MOVE_TIMEOUT_S)
  _report_state(robot, "newer motion finished")


def main(_):
  robot = r2client.Robot(
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=(
          f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_QUERY_PORT}"
      ),
      training_server_address=(
          f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}"
      ),
  )

  print("Column state at start")
  start = _report_state(robot, "start")
  if not start.connected:
    print(
        "\nThe backend has no link to the column. Check that the column"
        " service is up on the hardware box and that enable_column is true"
        " in the selected system config."
    )
    return

  _show_move_refused_while_uncalibrated(robot)
  _show_move_refused_while_calibrating(robot)
  _show_completed_move(robot)
  _show_stopped_move(robot)
  _show_cancelled_owner(robot)
  _show_cancelled_queued_motion(robot)
  _show_cancelled_superseded_motion(robot)

  print("\nDone.")


if __name__ == "__main__":
  app.run(main)
