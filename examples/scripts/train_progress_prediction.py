"""Train a progress prediction model from data warehouse entries.

This script trains a progress prediction model using the r2_labs SDK.
The model predicts task completion progress (0-1) from camera images,
used for behavior termination detection.

Usage:
    # Train from full episodes matching a pattern
    python train_progress_prediction.py \
        --entry_filters="pick_up_can*" \
        --training_steps=10000

    # Train from multiple entry filters
    python train_progress_prediction.py \
        --entry_filters="pick_up_can*,place_object*" \
        --training_steps=10000

    # Train from human demonstration segments (DAgger data)
    python train_progress_prediction.py \
        --human_entry_filters="dagger_*" \
        --training_steps=10000

    # Mix of full episodes and human demonstrations
    python train_progress_prediction.py \
        --entry_filters="agent_demos*" \
        --human_entry_filters="dagger_pick*,dagger_place*" \
        --training_steps=10000

The script automatically:
1. Builds a dataset from matching data warehouse entries
2. Caches the dataset for faster subsequent runs
3. Trains the progress prediction model
4. Exports the model to the model warehouse
"""

import time

from absl import app, flags

from r2_labs import client as r2client
from r2_labs import rpc_api

FLAGS = flags.FLAGS

# Entry filter configuration
flags.DEFINE_list(
    "entry_filters",
    [],
    "Comma-separated glob patterns for full episode entries from data warehouse "
    "(e.g., 'pick_up_can*,place_object*'). Processes entire episodes.",
)

flags.DEFINE_list(
    "human_entry_filters",
    [],
    "Comma-separated glob patterns for human demonstration entries from data "
    "warehouse (e.g., 'dagger_*'). Extracts only the human segments.",
)

# Training configuration
flags.DEFINE_string(
    "model_name",
    "progress_model",
    "Name for the exported model in the model warehouse.",
)

flags.DEFINE_integer(
    "training_steps",
    10_000,
    "Total number of training steps.",
)

flags.DEFINE_integer(
    "batch_size",
    32,
    "Training batch size.",
)

flags.DEFINE_bool(
    "force_rebuild",
    False,
    "If True, rebuild the dataset even if a cached version exists.",
)

# Task configuration
flags.DEFINE_enum(
    "task_type",
    "classification",
    ["classification", "regression"],
    "Task type: 'classification' for binary done/not-done prediction, "
    "'regression' for continuous 0-1 progress.",
)

# Camera configuration
flags.DEFINE_string(
    "camera",
    "wrist_camera",
    "Camera name to use (e.g., wrist_camera).",
)

# Server configuration
flags.DEFINE_string(
    "server_address",
    f"tcp://localhost:{rpc_api.DEFAULT_PORT}",
    "Main RPC server address.",
)


def main(_):
  # Filter out empty strings from lists
  entry_filters = [f for f in FLAGS.entry_filters if f]
  human_entry_filters = [f for f in FLAGS.human_entry_filters if f]

  if not entry_filters and not human_entry_filters:
    raise ValueError(
        "At least one of --entry_filters or --human_entry_filters is required"
    )

  # Connect to robot server
  robot = r2client.Robot(
      FLAGS.server_address,
      query_server_address=f"tcp://localhost:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=(
          f"tcp://localhost:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}"
      ),
  )

  # Start training
  print("Starting progress prediction training...")
  print(f"  Model name: {FLAGS.model_name}")
  print(f"  Entry filters: {entry_filters or '(none)'}")
  print(f"  Human entry filters: {human_entry_filters or '(none)'}")
  print(f"  Training steps: {FLAGS.training_steps}")
  print(f"  Batch size: {FLAGS.batch_size}")
  print(f"  Task type: {FLAGS.task_type}")
  print(f"  Camera: {FLAGS.camera}")

  response = robot.progress_trainer.train_model(
      model_name=FLAGS.model_name,
      training_steps=FLAGS.training_steps,
      entry_filters=entry_filters or None,
      human_entry_filters=human_entry_filters or None,
      force_rebuild=FLAGS.force_rebuild,
      batch_size=FLAGS.batch_size,
      task_type=FLAGS.task_type,
      cameras=[FLAGS.camera],
  )

  if response.error:
    print(f"Failed to start training: {response.error}")
    return

  if response.dataset_was_rebuilt:
    print(f"Built new dataset with {response.current_entry_count} entries")
  elif response.dataset_is_stale:
    print(
        f"WARNING: Using stale cached dataset "
        f"({response.cached_entry_count} entries)"
    )
    print(f"Current data has {response.current_entry_count} entries")
    print("Re-run with --force_rebuild for fresh data")
  else:
    print(f"Using cached dataset ({response.cached_entry_count} entries)")

  # Poll for training completion
  print("\nTraining in progress...")
  while True:
    status = robot.progress_trainer.get_training_status()
    if status.is_finished:
      break

    progress_pct = 100 * status.steps_completed / status.max_steps
    metrics = f"loss={status.loss:.4f}"
    if status.accuracy is not None:
      metrics += f", acc={status.accuracy:.4f}"
    if status.f1 is not None:
      metrics += f", f1={status.f1:.4f}"

    print(
        f"  Step {status.steps_completed}/{status.max_steps} "
        f"({progress_pct:.1f}%) - {metrics}"
    )
    time.sleep(10.0)

  print("\nTraining completed!")
  print(f"Model '{FLAGS.model_name}' exported to model warehouse")


if __name__ == "__main__":
  app.run(main)
