"""Shows what each force limit would stop among runs you have judged.

Give the data warehouse entry ids of visual trajectory runs you call good and,
optionally, of runs you call failures. The robot works out the force each run
applied from what it recorded. Prints, per limit, how many of each it stops,
then the lowest limit that stops none of the good runs, so you can pick the
max_applied_force to replay with. The arm does not move.
"""

import collections.abc
import dataclasses
import math

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "good", None, "Data warehouse entry ids of good runs", required=True
)

flags.DEFINE_list(
    "failures", [], "Data warehouse entry ids of runs that went wrong"
)

flags.DEFINE_string(
    "hostname",
    "localhost",
    "Hostname of the robot running the RPC API service.",
)


@dataclasses.dataclass(frozen=True)
class JudgedRun:
  recording: str
  motion_type: rpc_api.TrajectoryMotionType
  peak: float


@dataclasses.dataclass
class LimitRow:
  limit: float
  good_stopped: int
  failures_stopped: int


def limit_rows(
    good_peaks: collections.abc.Sequence[float],
    failure_peaks: collections.abc.Sequence[float],
) -> list[LimitRow]:
  """Limits from the one that stops no good run downwards.

  Each limit is the lowest that still lets one more good run through. The rows
  end once every failure is stopped, since lower limits only stop more good
  runs.
  """
  # A run is stopped above the limit, so one at the limit passes. Rounded up
  # to the 0.1 N the table prints.
  limits = {math.ceil(peak * 10) / 10 for peak in good_peaks}
  rows = []
  for limit in sorted(limits, reverse=True):
    row = LimitRow(
        limit,
        sum(peak > limit for peak in good_peaks),
        sum(peak > limit for peak in failure_peaks),
    )
    rows.append(row)
    if failure_peaks and row.failures_stopped == len(failure_peaks):
      break
  return rows


def judged_runs(
    query: r2client.QueryClient, entry_ids: collections.abc.Iterable[str]
) -> list[JudgedRun]:
  """The runs the robot has an applied force for; says why for the others."""
  runs = []
  for entry_id in entry_ids:
    run = query.get_recorded_applied_force(entry_id)
    if run.error is not None:
      print(f"Left out: {run.error}.")
      continue
    assert run.visual_trajectory_name is not None
    assert run.motion_type is not None
    assert run.applied_forces is not None
    runs.append(
        JudgedRun(
            run.visual_trajectory_name,
            run.motion_type,
            float(run.applied_forces.max()),
        )
    )
  return runs


def main(_):

  robot = r2client.Robot(
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  good = judged_runs(robot.query, FLAGS.good)
  failures = judged_runs(robot.query, FLAGS.failures)
  if not good:
    raise app.UsageError("The robot has an applied force for no good run.")
  kinds = {(run.recording, run.motion_type.name) for run in good + failures}
  if len(kinds) > 1:
    raise app.UsageError(
        f"The runs differ in recording or motion type: {sorted(kinds)}. Each"
        " needs its own limit."
    )

  good_peaks = [run.peak for run in good]
  failure_peaks = [run.peak for run in failures]
  print(
      f"{good[0].recording}: {len(good)} good runs, {len(failures)} failures."
  )
  print("   limit    good stopped  failures stopped")
  rows = limit_rows(good_peaks, failure_peaks)
  for row in rows:
    good_stopped = f"{row.good_stopped} of {len(good)}"
    failures_stopped = f"{row.failures_stopped} of {len(failures)}"
    print(f"{row.limit:6.1f} N  {good_stopped:>14}  {failures_stopped:>16}")
  print(
      f"Stops none of your good runs: --max_applied_force {rows[0].limit:.1f}"
  )


if __name__ == "__main__":
  app.run(main)
