"""Start an online behaviour-cloning run on a training server.

Two ways to use it:
  1. Edit the CONFIG block below and run with no arguments.
  2. Override any field on the command line (tyro), e.g. --model_name foo.

Connects to a running training server (start it with
`python -m bot01.sdk.server.run_training_server`) and starts an online BC
session: the trainer runs continuously on the growing dataset at
online_dataset_dir (episodes arrive live from the robot backend's forwarder)
and republishes the served model to online_model_dir every
snapshot_interval_steps for the inference service to hot-reload. Both
directories are on the training server. Warm-starting (init_from_model_id) is
required for a fresh run: without it the served startup snapshot is an
untrained policy.

Run by path (the examples dir is not an importable package). Example:
  uv run python r2_labs/examples/scripts/train_online_bc.py --host localhost --model_name online_run1 --monitor
"""

import dataclasses
import time

import dotenv
from loguru import logger as log
import tyro

from r2_labs.rpc import client as rpc_client
from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api

# ======================================================================
# CONFIG — edit these and run with no arguments (or override on the CLI).
# ======================================================================

# --- Server ---
HOST = "localhost"  # training server hostname/IP
PORT = rpc_api.DEFAULT_MODEL_TRAINER_PORT  # training server port (7534)

# --- Run identity + directories (all on the training server) ---
MODEL_NAME = "online_bc_run"  # name for checkpoints / ClearML
ONLINE_DATASET_DIR = "/data/r2/online_bc/grow"  # growing dataset dir
ONLINE_MODEL_DIR = "/data/r2/online_bc/serve"  # served-snapshot dir (watched)

# --- Warm start (required for a fresh run) ---
INIT_FROM_MODEL_ID = ""  # model warehouse id to warm-start weights from

# --- Data mixing (pretrained + online) ---
# When PRETRAINED_DATASET_DIR is set, each batch is drawn from a mixture of
# this static pretrained-data zarr (a finite trajectory dataset, cycled) and
# the growing ONLINE_DATASET_DIR, split PRETRAINED_WEIGHT / (1 - weight).
# Empty disables mixing (train on the online dataset alone).
PRETRAINED_DATASET_DIR = ""  # e.g. .../sdk_flow_matching_datasets/<task>/train
PRETRAINED_WEIGHT = 0.5  # fraction of each batch from the pretrained data

# --- Model / data shape (must match the warm-start model) ---
PREDICTION_HORIZON = 32  # future steps predicted
CAMERAS: tuple[str, ...] | None = None  # None = server default (wrist+right)
USE_JOINT_TORQUES = False  # include piper_joint_torques in proprio
BATCH_SIZE = 64

# --- Training schedule ---
TRAINING_STEPS = 1_000_000  # absolute step cap
SNAPSHOT_INTERVAL_STEPS = 1000  # steps between safetensors republishes
CHECKPOINT_INTERVAL_STEPS = 1000
MAX_CHECKPOINTS_TO_KEEP = 10
ONLINE_LEARNING_RATE: float | None = None  # None = server's default slow rate

# --- Collect-only ---
# Attach the episode exporter and append forwarded episodes to
# ONLINE_DATASET_DIR, but run NO training (no gradient steps, no snapshot
# republishing). For collecting rollouts into the growing dataset while
# evaluating a served policy. Warm start / pretrained mixing are irrelevant.
COLLECT_ONLY = False

# --- Misc ---
USE_ZERO_FALLBACK_FOR_MISSING_CAMERAS = False
TIMEOUT_MS = 30_000
MONITOR = False  # after starting, poll and print status until interrupted
POLL_INTERVAL = 5.0  # seconds between status polls when MONITOR is set

# ======================================================================


@dataclasses.dataclass
class Args:
  """Online BC training parameters (defaults come from the CONFIG block)."""

  model_name: str = MODEL_NAME
  online_dataset_dir: str = ONLINE_DATASET_DIR
  online_model_dir: str = ONLINE_MODEL_DIR
  init_from_model_id: str = INIT_FROM_MODEL_ID
  pretrained_dataset_dir: str = PRETRAINED_DATASET_DIR
  pretrained_weight: float = PRETRAINED_WEIGHT
  host: str = HOST
  port: int = PORT
  training_steps: int = TRAINING_STEPS
  snapshot_interval_steps: int = SNAPSHOT_INTERVAL_STEPS
  checkpoint_interval_steps: int = CHECKPOINT_INTERVAL_STEPS
  max_checkpoints_to_keep: int = MAX_CHECKPOINTS_TO_KEEP
  cameras: tuple[str, ...] | None = CAMERAS
  batch_size: int = BATCH_SIZE
  prediction_horizon: int = PREDICTION_HORIZON
  use_joint_torques: bool = USE_JOINT_TORQUES
  use_zero_fallback_for_missing_cameras: bool = (
      USE_ZERO_FALLBACK_FOR_MISSING_CAMERAS
  )
  online_learning_rate: float | None = ONLINE_LEARNING_RATE
  collect_only: bool = COLLECT_ONLY
  timeout_ms: int = TIMEOUT_MS
  monitor: bool = MONITOR
  poll_interval: float = POLL_INTERVAL


def _print_status(status: rpc_api.TrainingStatusResponse) -> None:
  log.info(
      "phase={} online_mode={} steps={}/{} loss={:.4f} fps={:.1f}",
      status.phase,
      status.online_mode,
      status.steps_completed,
      status.max_steps,
      status.loss,
      status.fps,
  )


def main(args: Args) -> None:
  server_address = f"tcp://{args.host}:{args.port}"
  log.info("Connecting to training server at {}", server_address)
  trainer = sdk_client.TrainerClient(
      rpc_client.BaseClient(
          server_address,
          timeout=args.timeout_ms,
          service_name="online training server",
      )
  )

  if not args.init_from_model_id and not args.collect_only:
    log.warning(
        "No init_from_model_id: the server rejects a fresh online run without"
        " a warm-start model (only a resume is exempt)."
    )

  config_overrides: dict[str, object] = {}
  if args.online_learning_rate is not None:
    config_overrides["online_learning_rate"] = args.online_learning_rate
  if args.pretrained_dataset_dir:
    # Mix pretrained + online per batch (default 50/50). Empty dir trains on
    # the online dataset alone.
    config_overrides["data.pretrained_dataset_dir"] = (
        args.pretrained_dataset_dir
    )
    config_overrides["data.pretrained_weight"] = args.pretrained_weight
    log.info(
        "Mixing pretrained data ({}) with online at weight {}",
        args.pretrained_dataset_dir,
        args.pretrained_weight,
    )

  response = trainer.start_online_training(
      model_name=args.model_name,
      training_steps=args.training_steps,
      online_dataset_dir=args.online_dataset_dir,
      online_model_dir=args.online_model_dir,
      init_from_model_id=args.init_from_model_id,
      snapshot_interval_steps=args.snapshot_interval_steps,
      checkpoint_interval_steps=args.checkpoint_interval_steps,
      max_checkpoints_to_keep=args.max_checkpoints_to_keep,
      cameras=list(args.cameras) if args.cameras is not None else None,
      batch_size=args.batch_size,
      prediction_horizon=args.prediction_horizon,
      use_joint_torques=args.use_joint_torques,
      use_zero_fallback_for_missing_cameras=(
          args.use_zero_fallback_for_missing_cameras
      ),
      config_overrides=config_overrides,
      collect_only=args.collect_only,
  )
  if response.error:
    log.error("Failed to start online training: {}", response.error)
    raise SystemExit(1)

  log.info(
      "Online training started: model={} dataset={} serve={}",
      args.model_name,
      args.online_dataset_dir,
      args.online_model_dir,
  )
  _print_status(trainer.get_online_training_status())

  if not args.monitor:
    log.info(
        "Detached. Monitor with get_online_training_status(); cancel with"
        " cancel_online_training()."
    )
    return

  log.info("Monitoring (Ctrl-C to detach; training keeps running)...")
  try:
    while True:
      time.sleep(args.poll_interval)
      status = trainer.get_online_training_status()
      _print_status(status)
      if status.is_finished:
        log.info("Online training finished (phase={}).", status.phase)
        break
  except KeyboardInterrupt:
    log.info("Detached from monitoring; training continues on the server.")


if __name__ == "__main__":
  dotenv.load_dotenv()
  main(tyro.cli(Args))
