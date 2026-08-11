"""Train an unconditioned skill model from data warehouse entries.

Raw entries must already be present on the training server.
The server builds and caches the prepared training dataset.

Example:
  uv run python r2_labs/examples/scripts/train_skill.py \
      --hostname diamond.local \
      --model_name insert_blue_square_uncond \
      --entry_filters 'insert_blue_square_into_top_center_hole#*' \
      --cameras wrist_camera \
      --training_steps 50000
"""

import time

from absl import app, flags
import dotenv
from loguru import logger as log

from r2_labs.rpc import client as rpc_client
from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "hostname",
    "localhost",
    "Hostname of the learned-skill training server.",
)
flags.DEFINE_string(
    "model_name",
    "offline_uncond_run",
    "Name for checkpoints and the exported model.",
)
flags.DEFINE_list(
    "entry_filters",
    [],
    "Comma-separated glob patterns matching data warehouse entry IDs.",
)
flags.DEFINE_list(
    "cameras",
    None,
    "Comma-separated camera names; omit to use the server defaults.",
)
flags.DEFINE_integer(
    "training_steps",
    50_000,
    "Number of optimizer steps.",
)


def _print_status(status: rpc_api.TrainingStatusResponse) -> None:
  if status.phase in ("preparing_dataset", "exporting_dataset"):
    log.info(
        "phase={} entries={}/{}",
        status.phase,
        status.export_entries_processed,
        status.export_entries_total,
    )
    return
  log.info(
      "phase={} steps={}/{} loss={:.4f} fps={:.1f}",
      status.phase,
      status.steps_completed,
      status.max_steps,
      status.loss,
      status.fps,
  )


def main(_: list[str]) -> None:
  dotenv.load_dotenv()
  if not FLAGS.entry_filters:
    raise ValueError("entry_filters must contain at least one glob pattern")

  server_address = (
      f"tcp://{FLAGS.hostname}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}"
  )
  log.info("Connecting to training server at {}", server_address)
  trainer = sdk_client.TrainerClient(
      rpc_client.BaseClient(
          server_address,
          timeout=30_000,
          service_name="skill training server",
      )
  )

  response = trainer.train_skill_model(
      model_name=FLAGS.model_name,
      model_family="unconditioned",
      training_steps=FLAGS.training_steps,
      entry_filters=FLAGS.entry_filters,
      cameras=FLAGS.cameras,
  )
  if response.error:
    raise RuntimeError(f"failed to start training: {response.error}")

  log.info(
      "Training started: model={} filters={}",
      FLAGS.model_name,
      FLAGS.entry_filters,
  )
  try:
    while True:
      status = trainer.get_training_status()
      _print_status(status)
      if status.phase == "failed":
        raise RuntimeError(status.error or "training failed")
      if status.is_finished:
        log.info("Training finished (phase={}).", status.phase)
        return
      time.sleep(5.0)
  except KeyboardInterrupt:
    log.info("Detached from monitoring; training continues on the server.")


if __name__ == "__main__":
  app.run(main)
