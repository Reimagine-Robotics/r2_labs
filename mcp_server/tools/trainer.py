"""Training and model management tools."""

import json

from mcp.server.fastmcp import Context

from r2_labs.mcp_server import serialization, server


@server.mcp.tool()
async def list_models(ctx: Context) -> str:
  """List all trained models."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.list_models)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def list_model_checkpoints(ctx: Context) -> str:
  """List model names available from training checkpoints."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.trainer.list_model_names_from_checkpoints
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_training_status(ctx: Context) -> str:
  """Get the current training status."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.get_training_status)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def is_training_running(ctx: Context) -> str:
  """Check if a training job is currently running."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.is_training_running)
  return json.dumps({"running": result})


@server.mcp.tool()
async def train_skill_model(
    model_name: str,
    training_steps: int,
    entry_filters: list[str],
    ctx: Context,
    entry_tags: list[str] | None = None,
    cameras: list[str] | None = None,
    batch_size: int = 64,
    prediction_horizon: int = 32,
    use_joint_torques: bool = False,
    checkpoint_interval_steps: int = 1000,
    max_checkpoints_to_keep: int = 10,
) -> str:
  """Start training a skill model.

  Args:
    model_name: name for the trained model.
    training_steps: number of training steps.
    entry_filters: list of trajectory entry names to train on.
    entry_tags: required data warehouse tags for entry filtering.
    cameras: camera names. None uses defaults; empty means no cameras.
    batch_size: training batch size.
    prediction_horizon: number of future steps the model predicts.
    use_joint_torques: whether to include joint torques as input.
    checkpoint_interval_steps: save a checkpoint every N steps.
    max_checkpoints_to_keep: maximum number of checkpoints to retain.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx,
      robot.trainer.train_skill_model,
      model_name=model_name,
      training_steps=training_steps,
      entry_filters=entry_filters,
      entry_tags=entry_tags,
      cameras=cameras,
      batch_size=batch_size,
      prediction_horizon=prediction_horizon,
      use_joint_torques=use_joint_torques,
      checkpoint_interval_steps=checkpoint_interval_steps,
      max_checkpoints_to_keep=max_checkpoints_to_keep,
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def cancel_training(ctx: Context) -> str:
  """Cancel the current training job."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.cancel_training)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def reset_trainer(ctx: Context) -> str:
  """Reset the trainer, stopping any running training and clearing state."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.reset_trainer)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def list_entry_filters(ctx: Context, search: str = "") -> str:
  """List available entry filters for training.

  Args:
    search: optional search string to filter results.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.trainer.list_entry_filters, search=search
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def start_export(
    ctx: Context,
    checkpoint_step: int | None = None,
    model_name: str | None = None,
    entry_filters: list[str] | None = None,
    entry_tags: list[str] | None = None,
    cameras: list[str] | None = None,
    model_save_dir: str | None = None,
    prediction_horizon: int | None = None,
    use_joint_torques: bool | None = None,
) -> str:
  """Start exporting a model from a training checkpoint.

  Args:
    checkpoint_step: export from this checkpoint step, or None for latest.
    model_name: optional model name override.
    entry_filters: optional entry filters override.
    entry_tags: required data warehouse tags. Must match training.
    cameras: camera names. Must match training.
    model_save_dir: optional save directory override.
    prediction_horizon: optional prediction horizon override.
    use_joint_torques: optional joint torques flag override.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx,
      robot.trainer.start_export,
      checkpoint_step=checkpoint_step,
      model_name=model_name,
      entry_filters=entry_filters,
      entry_tags=entry_tags,
      cameras=cameras,
      model_save_dir=model_save_dir,
      prediction_horizon=prediction_horizon,
      use_joint_torques=use_joint_torques,
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_export_status(ctx: Context) -> str:
  """Get the status of an ongoing or completed model export."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.get_export_status)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def list_checkpoints(ctx: Context) -> str:
  """List available training checkpoint steps."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trainer.list_checkpoints)
  return json.dumps(serialization.serialize(result))
