"""Move the column to a height above the calibrated bottom, in millimetres."""

import argparse

from r2_labs import client as r2client
from r2_labs import rpc_api


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Home the column if needed, then move. Keep people clear."
  )
  parser.add_argument(
      "height_mm", type=float, help="Height above bottom in mm."
  )
  parser.add_argument("--hostname", default="localhost", help="Robot hostname.")
  args = parser.parse_args()

  robot = r2client.Robot(
      f"tcp://{args.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=(
          f"tcp://{args.hostname}:{rpc_api.DEFAULT_QUERY_PORT}"
      ),
      training_server_address=(
          f"tcp://{args.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}"
      ),
  )
  motion = None
  try:
    if not robot.column.get_state().calibrated:
      print("Calibrating column. Keep people clear.")
      motion = robot.column.calibrate()
      motion.result()
    motion = robot.column.go_to(args.height_mm)
    motion.result()
  except KeyboardInterrupt:
    if motion is not None:
      motion.cancel()
    raise
  print(f"Column height {robot.column.get_state().height_mm:.1f} mm")


if __name__ == "__main__":
  main()
