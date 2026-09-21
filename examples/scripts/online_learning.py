r"""Start, follow, or stop an online-learning session from the shell.

The training server warm-starts from a model, serves it for the robot, and
trains continuously on the episodes the robot backend forwards, republishing
the served weights as it goes. This script starts that session, attaches the
robot's episode forwarding to it, and prints the trainer's status. The
session belongs to the server: Ctrl-C leaves the script, not the session,
which keeps training and receiving episodes. `--mode stop` ends it: it waits
for any episode in progress, detaches forwarding, so no episode is saved
under a session that has ended, and then cancels training, which exports the
final weights to the model warehouse.

Before starting: the training server must be running, the warm-start model
must be in its model warehouse, and the robot's system configuration must
name the training server in `online_learning_server_address`.

Example:
  uv run python r2_labs/examples/scripts/online_learning.py \
      --robot_hostname robot.local --training_server_hostname trainer.local \
      --model_id 'skill_18_08_07_00#calm-credits-28' \
      --cameras wrist_camera --use_joint_torques \
      --config_override optimizer.l2_toward_init=1.0

The server builds the session's training config from these arguments, not
from the warm-start model, so restate the model's architecture where it
differs from the defaults: cameras, prediction horizon, and joint torques
have flags; `model.*` fields go through --config_override. A warm start that
does not cover every parameter fails the session shortly after it starts,
and the status line reports the error.

Any training config field is set the same way, as `dotted.path=value` with
the value in JSON, so strings are quoted: `'model.dino_model_name="vits16"'`.

Rerunning with the same --model_id resumes the session in place; --restart
begins again from the warm-start weights with an empty online dataset,
keeping the previous state as a backup on the server. --mode attach rejoins a
session that is already training.
"""

import json
import time

from absl import app, flags
from loguru import logger as log

from r2_labs.rpc import client as rpc_client
from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "mode",
    "start",
    ["start", "attach", "stop"],
    "start a session and follow it; attach to and follow the session already"
    " training; or stop the current session.",
)
flags.DEFINE_string(
    "robot_hostname", "localhost", "Hostname of the robot backend."
)
flags.DEFINE_string(
    "training_server_hostname", "localhost", "Hostname of the training server."
)
flags.DEFINE_string(
    "model_id", None, "Model warehouse id to warm-start from (mode start)."
)
flags.DEFINE_integer(
    "inference_gpu", 0, "GPU on the training server for the served policy."
)
flags.DEFINE_integer(
    "inference_port",
    4243,
    "Port on the training server for the served policy; fixed so the robot's"
    " policy-service address is stable across sessions.",
)
flags.DEFINE_list(
    "cameras",
    None,
    "Comma-separated camera names the model was trained with; omit for the"
    " server defaults.",
)
flags.DEFINE_integer(
    "prediction_horizon", 32, "Action prediction horizon of the model."
)
flags.DEFINE_bool(
    "use_joint_torques",
    False,
    "Whether the model takes joint torques as proprioceptive input.",
)
flags.DEFINE_multi_string(
    "config_override",
    [],
    "Training config override as dotted.path=value with the value in JSON;"
    " repeatable.",
)
flags.DEFINE_bool(
    "restart",
    False,
    "Start again from the warm-start weights with an empty online dataset.",
)

_STATUS_INTERVAL_SECONDS = 10.0
# Starting a session spawns and compiles the inference service.
_START_TIMEOUT_MS = 240_000
# Status polls queue behind episode appends on the single-threaded server.
_RPC_TIMEOUT_MS = 30_000


def parse_config_overrides(pairs: list[str]) -> dict[str, object]:
  """Parse `dotted.path=value` pairs whose values are JSON."""
  overrides: dict[str, object] = {}
  for pair in pairs:
    key, sep, value = pair.partition("=")
    if not sep:
      raise app.UsageError(f"--config_override needs key=value, got {pair!r}")
    try:
      overrides[key] = json.loads(value)
    except json.JSONDecodeError as error:
      raise app.UsageError(
          f"--config_override {key} has a non-JSON value {value!r};"
          ' quote strings, e.g. name="value"'
      ) from error
  return overrides


def connect(
    robot_hostname: str, training_server_hostname: str
) -> sdk_client.Robot:
  """Connect to the robot backend and, through it, the training server."""
  return sdk_client.Robot(
      server_address=f"tcp://{robot_hostname}:{rpc_api.DEFAULT_PORT}",
      query_server_address=(
          f"tcp://{robot_hostname}:{rpc_api.DEFAULT_QUERY_PORT}"
      ),
      training_server_address=(
          f"tcp://{training_server_hostname}:"
          f"{rpc_api.DEFAULT_MODEL_TRAINER_PORT}"
      ),
      timeout=_RPC_TIMEOUT_MS,
  )


def start_session(robot: sdk_client.Robot) -> None:
  """Ask the trainer to start the session the start-mode flags describe.

  Overrides are parsed before anything is sent, so a malformed one fails
  before a session exists.
  """
  if FLAGS.model_id is None:
    raise app.UsageError("--model_id is required in mode start")
  config_overrides = parse_config_overrides(FLAGS.config_override)
  response = robot.trainer.start_online_learning(
      init_from_model_id=FLAGS.model_id,
      inference_gpu=FLAGS.inference_gpu,
      inference_port=FLAGS.inference_port,
      cameras=FLAGS.cameras,
      prediction_horizon=FLAGS.prediction_horizon,
      use_joint_torques=FLAGS.use_joint_torques,
      restart_online_learning=FLAGS.restart,
      config_overrides=config_overrides,
      timeout=_START_TIMEOUT_MS,
  )
  if response.error is not None:
    raise RuntimeError(f"Could not start the session: {response.error}")
  log.info(
      "Session {} started; policy served at {}.",
      response.online_learning_model_name,
      response.online_learning_inference_address,
  )


def attach_forwarding(robot: sdk_client.Robot) -> None:
  """Forward the robot's saved episodes to the session the trainer is running."""
  status = robot.trainer.get_online_learning_status()
  if status.is_finished or status.model_name is None:
    raise RuntimeError("No session is running.")
  forwarding = robot.online_episode_forwarding.start(status.model_name)
  if forwarding.error is not None:
    raise RuntimeError(f"Could not attach forwarding: {forwarding.error}")
  log.info("Forwarding attached to session {}.", status.model_name)


def follow_session(robot: sdk_client.Robot) -> rpc_api.TrainingStatusResponse:
  """Print the trainer's status until the session ends on the server.

  A timed-out poll is retried.
  """
  while True:
    try:
      status = robot.trainer.get_online_learning_status()
    except rpc_client.RpcTimeoutError:
      log.warning("Status poll timed out; retrying.")
      continue
    _print_status(status)
    if status.is_finished:
      return status
    time.sleep(_STATUS_INTERVAL_SECONDS)


def detach_forwarding(robot: sdk_client.Robot) -> None:
  """Detach the robot's forwarding once no episode is in progress."""
  if _episode_in_progress(robot):
    log.info("Waiting for the current episode to end before detaching.")
  while _episode_in_progress(robot):
    time.sleep(2.0)
  forwarding = robot.online_episode_forwarding.stop()
  if forwarding.error is not None:
    raise RuntimeError(f"Could not detach forwarding: {forwarding.error}")
  log.info("Forwarding detached.")


def stop_session(robot: sdk_client.Robot) -> None:
  """Detach forwarding, then cancel training; the final weights are exported."""
  log.info("Stopping the session.")
  detach_forwarding(robot)
  cancelled = robot.trainer.cancel_online_learning()
  if cancelled.error is not None:
    raise RuntimeError(f"Could not cancel the session: {cancelled.error}")
  log.info("Session cancelled; final weights exporting.")


def _print_status(status: rpc_api.TrainingStatusResponse) -> None:
  waiting = status.phase == "training" and status.steps_completed == 0
  waiting_note = " (waiting for the first saved episode)" if waiting else ""
  log.info(
      "phase={} steps={}/{} loss={:.4f} fps={:.1f}{}",
      status.phase,
      status.steps_completed,
      status.max_steps,
      status.loss,
      status.fps,
      waiting_note,
  )


def _episode_in_progress(robot: sdk_client.Robot) -> bool:
  state = robot.episode_observer.get_state()
  return state.is_recording or state.pending_save_decision


def main(_: list[str]) -> None:
  robot = connect(FLAGS.robot_hostname, FLAGS.training_server_hostname)
  if FLAGS.mode == "stop":
    stop_session(robot)
    return

  if FLAGS.mode == "start":
    start_session(robot)
  attach_forwarding(robot)
  log.info("Ctrl-C leaves the session running; --mode stop ends it.")
  try:
    status = follow_session(robot)
  except KeyboardInterrupt:
    log.info(
        "Session continues on the server; rerun with --mode stop to end it."
    )
    return

  # The server ended the session on its own, so there is nothing to cancel.
  detach_forwarding(robot)
  if status.phase == "failed":
    raise RuntimeError(f"The session failed: {status.error}")
  log.info("Session ended on the server (phase={}).", status.phase)


if __name__ == "__main__":
  app.run(main)
