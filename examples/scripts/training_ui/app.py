"""R2 Training Studio - FastAPI Backend."""

import asyncio
import json
import socket
import traceback
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from r2_labs.rpc import client as rpc_client
from r2_labs.sdk import client as sdk_client

app = FastAPI(title="R2 Training UI")

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global trainer clients (set via /connect endpoint)
trainer: sdk_client.TrainerClient | None = None  # Skill model trainer
progress_trainer: sdk_client.ProgressPredictionTrainerClient | None = (
    None  # Progress prediction trainer
)
server_address: str | None = None  # Store for hard reset


# Request/Response Models
class ConnectRequest(BaseModel):
  host: str
  port: int


class TrainRequest(BaseModel):
  model_name: str
  training_steps: int
  entry_filters: list[str]
  batch_size: int = 32
  prediction_horizon: int = 32
  force_rebuild: bool = False
  use_joint_torques: bool = False


@app.get("/")
async def root():
  """Serve the main UI."""
  index_path = static_dir / "index.html"
  return FileResponse(index_path)


@app.get("/api/server_info")
async def server_info():
  """Get info about the UI server itself."""
  hostname = socket.gethostname()
  return {
      "hostname": hostname,
      "port": 8000,
  }


@app.post("/api/connect")
async def connect(request: ConnectRequest):
  """Connect to the training server."""
  global trainer, progress_trainer, server_address

  try:
    server_addr = f"tcp://{request.host}:{request.port}"

    # Create base client (constructor pings server, so run in thread)
    base_client = await asyncio.to_thread(
        rpc_client.BaseClient, server_addr, timeout=5000
    )

    # Create both trainer clients
    test_trainer = sdk_client.TrainerClient(base_client)
    test_progress_trainer = sdk_client.ProgressPredictionTrainerClient(
        base_client
    )

    # Test connection with actual RPC call
    status = await asyncio.to_thread(test_trainer.get_training_status)

    # Connection successful - set global trainers and server address
    trainer = test_trainer
    progress_trainer = test_progress_trainer
    server_address = server_addr

    return {
        "success": True,
        "server": server_addr,
        "phase": status.phase,
    }
  except Exception as e:
    # Reset trainer on failure
    trainer = None
    error_msg = str(e)

    # Provide more helpful error messages
    if "Connection refused" in error_msg or "Failed to connect" in error_msg:
      error_msg = f"Cannot connect to {request.host}:{request.port}. Is the training server running?"
    elif "timeout" in error_msg.lower():
      error_msg = f"Connection timeout to {request.host}:{request.port}. Server not responding."

    return {"success": False, "error": error_msg}


@app.post("/api/disconnect")
async def disconnect():
  """Disconnect from the training server."""
  global trainer, progress_trainer, server_address
  trainer = None
  progress_trainer = None
  server_address = None
  return {"success": True}


@app.post("/api/hard_reset")
async def hard_reset():
  """Hard reset - destroy trainer and create a fresh connection."""
  global trainer, progress_trainer, server_address

  if (trainer is None and progress_trainer is None) or server_address is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    # Store server address before destroying trainer
    server_addr = server_address

    # Reset both trainers on server side (cancels training and clears state)
    print("[Hard Reset] Resetting trainers on server...")

    # Reset flow matching trainer
    try:
      if trainer:
        reset_response = await asyncio.to_thread(
            trainer.reset_trainer  # type: ignore
        )
        if not reset_response.success:
          print(
              f"[Hard Reset] Flow matching reset failed: {reset_response.error}"
          )
        else:
          print("[Hard Reset] Flow matching trainer reset successful")
    except Exception as e:
      print(f"[Hard Reset] Flow matching reset failed: {e}")

    # Reset progress prediction trainer
    try:
      if progress_trainer:
        reset_response = await asyncio.to_thread(
            progress_trainer.reset_trainer  # type: ignore
        )
        if not reset_response.success:
          print(
              f"[Hard Reset] Progress trainer reset failed: {reset_response.error}"
          )
        else:
          print("[Hard Reset] Progress trainer reset successful")
    except Exception as e:
      print(f"[Hard Reset] Progress trainer reset failed: {e}")

    # Destroy old trainers (close connections)
    old_trainer = trainer
    old_progress_trainer = progress_trainer
    trainer = None
    progress_trainer = None
    del old_trainer
    del old_progress_trainer

    # Create fresh connections
    print(f"[Hard Reset] Creating fresh trainer connections to {server_addr}")
    base_client = await asyncio.to_thread(
        rpc_client.BaseClient, server_addr, timeout=5000
    )
    new_trainer = sdk_client.TrainerClient(base_client)
    new_progress_trainer = sdk_client.ProgressPredictionTrainerClient(
        base_client
    )

    # Test connections
    status = await asyncio.to_thread(new_trainer.get_training_status)
    progress_status = await asyncio.to_thread(
        new_progress_trainer.get_training_status
    )

    # Set new trainers
    trainer = new_trainer
    progress_trainer = new_progress_trainer

    print(
        f"[Hard Reset] Success - skill phase={status.phase}, progress phase={progress_status.phase}"
    )

    return {
        "success": True,
        "server": server_addr,
        "phase": status.phase,
    }
  except Exception as e:
    trainer = None
    server_address = None
    return {"success": False, "error": str(e)}


@app.get("/api/status")
async def get_status():
  """Get current training status."""
  if trainer is None:
    return {"connected": False, "phase": "idle"}

  try:
    status = await asyncio.to_thread(
        trainer.get_training_status  # type: ignore
    )
    return {
        "connected": True,
        "phase": status.phase,
        "is_finished": status.is_finished,
        "steps_completed": status.steps_completed,
        "max_steps": status.max_steps,
    }
  except Exception as e:
    return {"connected": False, "error": str(e), "phase": "idle"}


@app.get("/api/list_models")
async def list_models():
  """List all exported models via training server RPC."""
  if trainer is None:
    return {"success": False, "error": "Not connected to server", "models": []}

  try:
    models = await asyncio.to_thread(trainer.list_models)  # type: ignore
    return {"success": True, "models": models}
  except Exception as e:
    print(f"[List Models] Error: {e}")
    traceback.print_exc()
    return {"success": False, "error": str(e), "models": []}


@app.get("/api/checkpoint_names")
async def get_checkpoint_names(search: str = "", prefix: str = ""):
  """List available model names from checkpoints via training server.

  Args:
    search: Filter by search string (case-insensitive).
    prefix: Filter to only show models starting with this prefix
            (e.g., 'rectify_skill_' or 'rectify_progress_').
  """
  if trainer is None:
    return {"success": False, "names": []}

  try:
    all_names = await asyncio.to_thread(
        trainer.list_model_names_from_checkpoints  # type: ignore
    )
    # Filter by prefix first (for separating skill vs progress models)
    if prefix:
      filtered = [n for n in all_names if n.startswith(prefix)]
    else:
      filtered = all_names
    # Then filter by search
    if search:
      filtered = [n for n in filtered if search.lower() in n.lower()]
    return {"success": True, "names": filtered[:50]}
  except Exception as e:
    print(f"[Checkpoint Names] Error: {e}")
    return {"success": False, "error": str(e), "names": []}


@app.get("/api/entry_filters")
async def get_entry_filters(search: str = ""):
  """Get available entry filter IDs from the data warehouse via RPC.

  Args:
      search: Optional search term to filter results.

  Returns:
      List of unique entry_filter_ids matching the search term.
  """
  if trainer is None:
    return {"success": False, "error": "Not connected to server", "filters": []}

  try:
    response = await asyncio.to_thread(
        trainer.list_entry_filters, search=search
    )
    return {
        "success": response.success,
        "filters": response.filters,
        "error": response.error,
    }
  except Exception as e:
    return {"success": False, "error": str(e), "filters": []}


@app.get("/api/training_status")
async def get_training_status():
  """Get current skill training status (for chat mode polling)."""
  if trainer is None:
    return {"connected": False, "error": "Not connected to server"}

  try:
    status = await asyncio.to_thread(
        trainer.get_training_status  # type: ignore
    )
    return {
        "connected": True,
        "phase": status.phase,
        "steps_completed": status.steps_completed,
        "max_steps": status.max_steps,
        "loss": status.loss if status.loss is not None else None,
        "fps": status.fps if status.fps is not None else None,
        "model_name": status.model_name,
        "entry_filters": status.entry_filters,
        "export_entries_processed": status.export_entries_processed,
        "export_entries_total": status.export_entries_total,
    }
  except Exception as e:
    return {"connected": False, "error": str(e)}


@app.get("/api/progress_training_status")
async def get_progress_training_status():
  """Get current progress prediction training status (for chat mode polling)."""
  if progress_trainer is None:
    return {"connected": False, "error": "Not connected to server"}

  try:
    status = await asyncio.to_thread(
        progress_trainer.get_training_status  # type: ignore
    )
    return {
        "connected": True,
        "phase": status.phase,
        "steps_completed": status.steps_completed,
        "max_steps": status.max_steps,
        "loss": status.loss if status.loss is not None else None,
        "fps": status.fps if status.fps is not None else None,
        "accuracy": status.accuracy if status.accuracy is not None else None,
        "model_name": status.model_name,
        "entry_filters": status.entry_filters,
        "export_entries_processed": status.export_entries_processed,
        "export_entries_total": status.export_entries_total,
    }
  except Exception as e:
    return {"connected": False, "error": str(e)}


@app.post("/api/train")
async def start_training(request: TrainRequest):
  """Start training."""
  if trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    response = await asyncio.to_thread(
        trainer.train_skill_model,  # type: ignore
        model_name=request.model_name,
        training_steps=request.training_steps,
        entry_filters=request.entry_filters,
        batch_size=request.batch_size,
        prediction_horizon=request.prediction_horizon,
        use_joint_torques=request.use_joint_torques,
        force_rebuild=request.force_rebuild,
        timeout=600000,
    )

    if response.error:
      return {"success": False, "error": response.error}
    return {"success": True}
  except Exception as e:
    return {"success": False, "error": str(e)}


@app.post("/api/cancel")
async def cancel_training():
  """Cancel training."""
  if trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    response = await asyncio.to_thread(trainer.cancel_training)  # type: ignore
    return {
        "success": response.success,
        "error": response.error,
    }
  except Exception as e:
    return {"success": False, "error": str(e)}


@app.post("/api/export")
async def export_model():
  """Export the current model (async operation)."""
  if trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    # Start async export
    print("[Export] Starting export...")
    response = await asyncio.to_thread(
        trainer.start_export, checkpoint_step=None  # type: ignore
    )
    if response.error:
      print(f"[Export] Start failed: {response.error}")
      return {"success": False, "error": response.error}

    print("[Export] Polling for completion...")
    # Poll for completion (max 60 seconds)
    for i in range(60):
      status = await asyncio.to_thread(
          trainer.get_export_status  # type: ignore
      )
      print(
          f"[Export] Poll {i+1}/60 - Finished: {status.is_finished}, Error: {status.error}"
      )
      if status.is_finished:
        if status.error:
          print(f"[Export] Failed: {status.error}")
          return {"success": False, "error": status.error}
        print(f"[Export] Success! Model ID: {status.model_id}")
        return {
            "success": True,
            "model_id": status.model_id,
            "checkpoint_step": status.checkpoint_step,
        }
      await asyncio.sleep(1)  # Use async sleep instead of blocking

    print("[Export] Timeout after 60 seconds")
    return {"success": False, "error": "Export timeout after 60 seconds"}

  except Exception as e:
    print(f"[Export] Exception: {e}")
    traceback.print_exc()
    return {"success": False, "error": str(e)}


# ============================================================================
# Progress Prediction Training Endpoints
# ============================================================================


@app.post("/api/progress/train")
async def start_progress_training(request: dict):
  """Start progress prediction training."""
  if progress_trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    # Get selected cameras
    cameras = request.get("cameras", ["wrist_camera", "right_camera"])

    # Get filters - at least one of entry_filters or human_entry_filters required
    entry_filters = request.get("entry_filters") or None
    human_entry_filters = request.get("human_entry_filters") or None

    if not entry_filters and not human_entry_filters:
      return {
          "success": False,
          "error": "At least one of entry_filters or human_entry_filters is required",
      }

    response = await asyncio.to_thread(
        progress_trainer.train_model,  # type: ignore
        model_name=request["model_name"],
        training_steps=request["training_steps"],
        entry_filters=entry_filters,
        human_entry_filters=human_entry_filters,
        batch_size=request.get("batch_size", 32),
        task_type=request.get("task_type", "classification"),
        cameras=cameras,
        force_rebuild=request.get("force_rebuild", False),
        checkpoint_interval_steps=request.get(
            "checkpoint_interval_steps", 1000
        ),
        max_checkpoints_to_keep=request.get("max_checkpoints_to_keep", 10),
    )

    if response.error:
      return {"success": False, "error": response.error}

    return {"success": True}

  except Exception as e:
    traceback.print_exc()
    return {"success": False, "error": str(e)}


@app.post("/api/progress/cancel")
async def cancel_progress_training():
  """Cancel progress prediction training."""
  if progress_trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    response = await asyncio.to_thread(
        progress_trainer.cancel_training  # type: ignore
    )
    return {
        "success": response.success,
        "error": response.error,
    }
  except Exception as e:
    return {"success": False, "error": str(e)}


@app.get("/api/progress/status")
async def get_progress_status():
  """Get progress prediction training status."""
  if progress_trainer is None:
    return {"connected": False, "phase": "idle"}

  try:
    status = await asyncio.to_thread(
        progress_trainer.get_training_status  # type: ignore
    )
    return {
        "connected": True,
        "phase": status.phase,
        "is_finished": status.is_finished,
        "steps_completed": status.steps_completed,
        "max_steps": status.max_steps,
        "loss": status.loss if status.loss else None,
        "accuracy": status.accuracy if status.accuracy else None,
        "f1": status.f1 if status.f1 else None,
        "fps": status.fps if status.fps else None,
        "val_loss": status.val_loss if status.val_loss else None,
        "val_accuracy": status.val_accuracy if status.val_accuracy else None,
        "val_f1": status.val_f1 if status.val_f1 else None,
        "checkpoint_id": status.checkpoint_id if status.checkpoint_id else None,
    }
  except Exception as e:
    return {"connected": False, "error": str(e), "phase": "idle"}


@app.get("/api/progress/checkpoints")
async def list_progress_checkpoints():
  """List available checkpoints for progress prediction model."""
  if progress_trainer is None:
    return {
        "success": False,
        "error": "Not connected to server",
        "checkpoints": [],
    }

  try:
    response = await asyncio.to_thread(
        progress_trainer.list_checkpoints  # type: ignore
    )
    return {"success": True, "checkpoints": response.checkpoint_steps}
  except Exception as e:
    print(f"[Progress Checkpoints] Error: {e}")
    return {"success": False, "error": str(e), "checkpoints": []}


@app.post("/api/progress/export")
async def export_progress_model(request: dict = Body(default={})):
  """Export the progress prediction model (async operation)."""
  if progress_trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    checkpoint_step = request.get("checkpoint_step")

    # Start async export
    print(
        f"[Progress Export] Starting export from checkpoint: {checkpoint_step}"
    )
    response = await asyncio.to_thread(
        progress_trainer.start_export,  # type: ignore
        checkpoint_step=checkpoint_step,
    )
    if response.error:
      print(f"[Progress Export] Start failed: {response.error}")
      return {"success": False, "error": response.error}

    print("[Progress Export] Polling for completion...")
    # Poll for completion (max 60 seconds)
    for i in range(60):
      status = await asyncio.to_thread(
          progress_trainer.get_export_status  # type: ignore
      )
      print(
          f"[Progress Export] Poll {i+1}/60 - Finished: {status.is_finished}, Error: {status.error}"
      )
      if status.is_finished:
        if status.error:
          print(f"[Progress Export] Failed: {status.error}")
          return {"success": False, "error": status.error}
        print(f"[Progress Export] Success! Model ID: {status.model_id}")
        return {
            "success": True,
            "model_id": status.model_id,
            "checkpoint_step": status.checkpoint_step,
        }
      await asyncio.sleep(1)

    print("[Progress Export] Timeout after 60 seconds")
    return {"success": False, "error": "Export timeout after 60 seconds"}

  except Exception as e:
    print(f"[Progress Export] Exception: {e}")
    traceback.print_exc()
    return {"success": False, "error": str(e)}


@app.post("/api/progress/reset")
async def reset_progress_trainer():
  """Reset progress prediction trainer to initial state."""
  if progress_trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    response = await asyncio.to_thread(
        progress_trainer.reset_trainer  # type: ignore
    )
    return {"success": response.success, "error": response.error}
  except Exception as e:
    print(f"[Progress Reset] Exception: {e}")
    traceback.print_exc()
    return {"success": False, "error": str(e)}


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
  """WebSocket endpoint for live training status updates."""
  print("[WS] WebSocket upgrade received, accepting...", flush=True)
  await websocket.accept()
  print("[WS] WebSocket accepted!", flush=True)

  try:
    while True:
      if trainer is None:
        await websocket.send_json(
            {"connected": False, "error": "Not connected to server"}
        )
        await asyncio.sleep(1)
        continue

      try:
        status = await asyncio.to_thread(
            trainer.get_training_status  # type: ignore
        )

        await websocket.send_json(
            {
                "connected": True,
                "phase": status.phase,
                "is_finished": status.is_finished,
                "steps_completed": status.steps_completed,
                "max_steps": status.max_steps,
                "loss": status.loss if status.loss is not None else None,
                "fps": status.fps if status.fps is not None else None,
                "export_entries_processed": status.export_entries_processed,
                "export_entries_total": status.export_entries_total,
                "metrics": status.metrics,
                # Config fields for auto-fill
                "model_name": status.model_name,
                "entry_filters": status.entry_filters,
                "batch_size": status.batch_size,
                "prediction_horizon": status.prediction_horizon,
            }
        )
      except Exception as e:
        print(f"[WebSocket] Error polling status: {e}")
        await websocket.send_json({"connected": False, "error": str(e)})

      await asyncio.sleep(1)  # Poll every second

  except WebSocketDisconnect:
    pass


# ============================================
# Claude AI Chat Integration
# ============================================


class ClaudeChatRequest(BaseModel):
  api_key: str
  messages: list[dict[str, Any]]
  context: dict[str, Any] | None = None


# Tool definitions for Claude
CLAUDE_TOOLS = [
    {
        "name": "start_skill_training",
        "description": "Start training a skill model (R2-M0 flow matching model). Use this when the user wants to train a new skill model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name for the model (will be prefixed with 'rectify_skill_' automatically if not present)",
                },
                "entry_filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entry filter patterns (e.g., ['rectify_*', 'pick_up_*'])",
                },
                "training_steps": {
                    "type": "integer",
                    "description": "Number of training steps (default: 40000)",
                    "default": 40000,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size (default: 32)",
                    "default": 32,
                },
                "prediction_horizon": {
                    "type": "integer",
                    "description": "Prediction horizon (default: 32)",
                    "default": 32,
                },
                "use_joint_torques": {
                    "type": "boolean",
                    "description": "Include piper_joint_torques in proprioception input (default: false)",
                    "default": False,
                },
                "force_rebuild": {
                    "type": "boolean",
                    "description": "Force rebuild the dataset even if cached",
                    "default": False,
                },
            },
            "required": ["model_name", "entry_filters"],
        },
    },
    {
        "name": "start_progress_training",
        "description": "Start training a progress prediction model. Use this when the user wants to train a progress predictor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name for the model (will be prefixed with 'rectify_progress_' automatically if not present)",
                },
                "entry_filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entry filter patterns",
                },
                "training_steps": {
                    "type": "integer",
                    "description": "Number of training steps (default: 10000)",
                    "default": 10000,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size (default: 32)",
                    "default": 32,
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "description": "Task type (default: classification)",
                    "default": "classification",
                },
                "force_rebuild": {
                    "type": "boolean",
                    "description": "Force rebuild the dataset even if cached",
                    "default": False,
                },
            },
            "required": ["model_name", "entry_filters"],
        },
    },
    {
        "name": "cancel_training",
        "description": "Cancel the currently running training. Use this when the user wants to stop training.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trainer_type": {
                    "type": "string",
                    "enum": ["skill", "progress"],
                    "description": "Which trainer to cancel (skill or progress). If not specified, will try to determine from context.",
                }
            },
        },
    },
    {
        "name": "export_model",
        "description": "Export the trained model to the model warehouse. Use this when the user wants to save/export their model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trainer_type": {
                    "type": "string",
                    "enum": ["skill", "progress"],
                    "description": "Which trainer to export from (skill or progress)",
                },
                "checkpoint_step": {
                    "type": "integer",
                    "description": "Specific checkpoint step to export (optional, uses latest if not specified)",
                },
            },
        },
    },
    {
        "name": "get_training_status",
        "description": "Get the current training status with live visualization. Use this when the user asks to see training progress, check status, or wants to see the loss chart. This will display an embedded status card with metrics and graph.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trainer_type": {
                    "type": "string",
                    "enum": ["skill", "progress", "both"],
                    "description": "Which trainer status to get",
                    "default": "both",
                },
                "show_visualization": {
                    "type": "boolean",
                    "description": "Whether to show a live visualization card (default: true)",
                    "default": True,
                },
            },
        },
    },
    {
        "name": "list_models",
        "description": "List available exported models from the model warehouse.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "hard_reset",
        "description": "Reset the trainer to initial state. Use this when the user wants to start fresh or clear errors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "trainer_type": {
                    "type": "string",
                    "enum": ["skill", "progress", "both"],
                    "description": "Which trainer to reset",
                    "default": "both",
                }
            },
        },
    },
]


async def execute_tool(tool_name: str, tool_input: dict) -> dict:
  """Execute a tool and return the result."""
  global trainer, progress_trainer

  try:
    if tool_name == "start_skill_training":
      if trainer is None:
        return {"error": "Not connected to training server"}

      model_name = tool_input["model_name"]
      if not model_name.startswith("rectify_skill_"):
        model_name = "rectify_skill_" + model_name

      entry_filters = tool_input["entry_filters"]
      # Add wildcard if not present
      entry_filters = [
          f + "*" if not f.endswith("*") else f for f in entry_filters
      ]

      result = await asyncio.to_thread(
          trainer.train_skill_model,  # type: ignore
          model_name=model_name,
          entry_filters=entry_filters,
          training_steps=tool_input.get("training_steps", 40000),
          batch_size=tool_input.get("batch_size", 32),
          prediction_horizon=tool_input.get("prediction_horizon", 32),
          use_joint_torques=tool_input.get("use_joint_torques", False),
          force_rebuild=tool_input.get("force_rebuild", False),
      )
      if result.error:
        return {"error": result.error}
      return {
          "success": True,
          "message": f"Started skill training for model '{model_name}'",
      }

    elif tool_name == "start_progress_training":
      if progress_trainer is None:
        return {"error": "Not connected to training server"}

      model_name = tool_input["model_name"]
      if not model_name.startswith("rectify_progress_"):
        model_name = "rectify_progress_" + model_name

      entry_filters = tool_input["entry_filters"]
      entry_filters = [
          f + "*" if not f.endswith("*") else f for f in entry_filters
      ]

      result = await asyncio.to_thread(
          progress_trainer.train_model,  # type: ignore
          model_name=model_name,
          entry_filters=entry_filters,
          training_steps=tool_input.get("training_steps", 10000),
          batch_size=tool_input.get("batch_size", 32),
          task_type=tool_input.get("task_type", "classification"),
          cameras=["wrist_camera", "right_camera"],
          force_rebuild=tool_input.get("force_rebuild", False),
      )
      if result.error:
        return {"error": result.error}
      return {
          "success": True,
          "message": f"Started progress training for model '{model_name}'",
      }

    elif tool_name == "cancel_training":
      trainer_type = tool_input.get("trainer_type", "skill")
      if trainer_type == "skill" and trainer:
        await asyncio.to_thread(trainer.cancel_training)  # type: ignore
        return {"success": True, "message": "Cancelled skill training"}
      elif trainer_type == "progress" and progress_trainer:
        await asyncio.to_thread(progress_trainer.cancel_training)  # type: ignore
        return {"success": True, "message": "Cancelled progress training"}
      return {"error": f"Trainer '{trainer_type}' not available"}

    elif tool_name == "export_model":
      trainer_type = tool_input.get("trainer_type", "skill")
      checkpoint_step = tool_input.get("checkpoint_step")

      if trainer_type == "skill" and trainer:
        result = await asyncio.to_thread(
            trainer.start_export, checkpoint_step=checkpoint_step  # type: ignore
        )
        if result.error:
          return {"error": result.error}
        return {"success": True, "message": "Started skill model export"}
      elif trainer_type == "progress" and progress_trainer:
        result = await asyncio.to_thread(
            progress_trainer.start_export, checkpoint_step=checkpoint_step  # type: ignore
        )
        if result.error:
          return {"error": result.error}
        return {"success": True, "message": "Started progress model export"}
      return {"error": f"Trainer '{trainer_type}' not available"}

    elif tool_name == "get_training_status":
      trainer_type = tool_input.get("trainer_type", "both")
      show_viz = tool_input.get("show_visualization", True)
      status_info = {}
      active_status = None

      if trainer_type in ["skill", "both"] and trainer:
        status = await asyncio.to_thread(
            trainer.get_training_status  # type: ignore
        )
        is_running = status.phase not in ("idle", "finished", "failed")
        skill_status = {
            "phase": status.phase,
            "is_running": is_running,
            "steps_completed": status.steps_completed,
            "max_steps": status.max_steps,
            "loss": status.loss,
            "fps": status.fps if hasattr(status, "fps") else None,
            "model_name": status.model_name,
            "trainer_type": "skill",
            "export_entries_processed": getattr(
                status, "export_entries_processed", 0
            ),
            "export_entries_total": getattr(status, "export_entries_total", 0),
        }
        status_info["skill"] = skill_status
        if is_running or status.phase in ("finished", "failed"):
          active_status = skill_status

      if trainer_type in ["progress", "both"] and progress_trainer:
        status = await asyncio.to_thread(
            progress_trainer.get_training_status  # type: ignore
        )
        is_running = status.phase not in ("idle", "finished", "failed")
        progress_status = {
            "phase": status.phase,
            "is_running": is_running,
            "steps_completed": status.steps_completed,
            "max_steps": status.max_steps,
            "loss": status.loss,
            "fps": status.fps if hasattr(status, "fps") else None,
            "accuracy": status.accuracy,
            "model_name": status.model_name,
            "trainer_type": "progress",
            "export_entries_processed": getattr(
                status, "export_entries_processed", 0
            ),
            "export_entries_total": getattr(status, "export_entries_total", 0),
        }
        status_info["progress"] = progress_status
        if (
            is_running or status.phase in ("finished", "failed")
        ) and not active_status:
          active_status = progress_status

      # Generate a descriptive message
      if active_status:
        phase = active_status.get("phase", "unknown")
        if phase == "training":
          message = f"Training in progress: {active_status.get('steps_completed', 0)}/{active_status.get('max_steps', 0)} steps"
        elif phase == "preparing_dataset":
          message = "Preparing dataset for training..."
        elif phase == "exporting_dataset":
          processed = active_status.get("export_entries_processed", 0)
          total = active_status.get("export_entries_total", 0)
          message = f"Exporting dataset: {processed}/{total} entries"
        elif phase == "finished":
          message = "Training finished successfully."
        elif phase == "failed":
          message = "Training failed."
        else:
          message = f"Training status: {phase}"
      else:
        message = "No active training session. Both trainers are idle."

      return {
          "success": True,
          "status": status_info,
          "show_visualization": show_viz,
          "active_status": active_status,  # For frontend to render status card
          "message": message,
      }

    elif tool_name == "list_models":
      if trainer is None:
        return {"error": "Not connected to training server"}
      models = await asyncio.to_thread(trainer.list_models)  # type: ignore
      return {
          "success": True,
          "models": models[:10],
      }  # Limit to 10 for chat display

    elif tool_name == "hard_reset":
      trainer_type = tool_input.get("trainer_type", "both")
      results = []

      if trainer_type in ["skill", "both"] and trainer:
        result = await asyncio.to_thread(trainer.reset_trainer)  # type: ignore
        results.append(
            f"Skill trainer: {'reset' if result.success else result.error}"
        )

      if trainer_type in ["progress", "both"] and progress_trainer:
        result = await asyncio.to_thread(
            progress_trainer.reset_trainer  # type: ignore
        )
        results.append(
            f"Progress trainer: {'reset' if result.success else result.error}"
        )

      return {"success": True, "message": "; ".join(results)}

    else:
      return {"error": f"Unknown tool: {tool_name}"}

  except Exception as e:
    return {"error": str(e)}


@app.post("/api/claude/chat")
async def claude_chat(request: ClaudeChatRequest):
  """Proxy Claude API requests with tool use support."""
  api_key = request.api_key
  messages = request.messages
  context = request.context or {}

  # Build system prompt
  system_prompt = """You are a helpful assistant for the R2 Training Studio.
You help users train robot skill models and progress prediction models.

Current context:
- Connected to server: {connected}
- Active trainer view: {current_trainer}
- Skill training status: {skill_status}
- Progress training status: {progress_status}

IMPORTANT: The status objects above are JSON and contain these fields from the trainer state:
- model_name: the current/last trained model name
- entry_filters: array of entry filter patterns used for training
- phase: current training phase (idle, preparing_dataset, training, finished, failed)

When users ask to "resume", "continue", "restart", or "train again" with the same settings:
1. Parse the skill_status or progress_status JSON to extract model_name and entry_filters
2. Use those exact values - DO NOT say you don't have access to them
3. If you see a JSON object with model_name and entry_filters fields, USE those values

When users ask to start training, use the appropriate tool.
When they mention model names without the prefix, add it automatically:
- Skill models: prefix with 'rectify_skill_'
- Progress models: prefix with 'rectify_progress_'

When specifying entry filters, add a wildcard '*' if not present.

Be concise but helpful. Confirm actions taken and report any errors clearly.""".format(
      connected=context.get("connected", False),
      current_trainer=context.get("currentTrainer", "none"),
      skill_status=json.dumps(context.get("skillStatus"))
      if context.get("skillStatus")
      else "idle",
      progress_status=json.dumps(context.get("progressStatus"))
      if context.get("progressStatus")
      else "idle",
  )

  # Convert messages to Claude format
  claude_messages = []
  for msg in messages:
    if msg["role"] == "user":
      claude_messages.append({"role": "user", "content": msg["content"]})
    elif msg["role"] == "assistant":
      claude_messages.append({"role": "assistant", "content": msg["content"]})
    # Skip system messages in history

  try:
    async with httpx.AsyncClient(timeout=60.0) as client:
      # Initial Claude API call
      response = await client.post(
          "https://api.anthropic.com/v1/messages",
          headers={
              "x-api-key": api_key,
              "anthropic-version": "2023-06-01",
              "content-type": "application/json",
          },
          json={
              "model": "claude-sonnet-4-20250514",
              "max_tokens": 1024,
              "system": system_prompt,
              "messages": claude_messages,
              "tools": CLAUDE_TOOLS,
          },
      )

      if response.status_code != 200:
        error_data = response.json()
        return {
            "success": False,
            "error": error_data.get("error", {}).get("message", "API error"),
        }

      data = response.json()

      # Check if Claude wants to use a tool
      tool_calls = []
      text_response = ""

      for content in data.get("content", []):
        if content["type"] == "text":
          text_response += content["text"]
        elif content["type"] == "tool_use":
          tool_name = content["name"]
          tool_input = content["input"]
          tool_id = content["id"]

          # Execute the tool
          tool_result = await execute_tool(tool_name, tool_input)
          tool_calls.append(
              {
                  "name": tool_name,
                  "result": tool_result.get("message")
                  or tool_result.get("error")
                  or "Done",
                  "full_result": tool_result,  # Include full result for visualization handling
              }
          )

          # If tool was used, make a follow-up call to get Claude's response
          if data.get("stop_reason") == "tool_use":
            # Add tool result to messages
            follow_up_messages = claude_messages + [
                {"role": "assistant", "content": data["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(tool_result),
                        }
                    ],
                },
            ]

            # Get Claude's final response
            follow_up = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": follow_up_messages,
                    "tools": CLAUDE_TOOLS,
                },
            )

            if follow_up.status_code == 200:
              follow_up_data = follow_up.json()
              for content in follow_up_data.get("content", []):
                if content["type"] == "text":
                  text_response = content["text"]

      return {
          "success": True,
          "response": text_response or "I've completed the action.",
          "tool_calls": tool_calls,
          "status_update": tool_calls[0]["result"] if tool_calls else None,
      }

  except httpx.TimeoutException:
    return {"success": False, "error": "Request timed out"}
  except Exception as e:
    traceback.print_exc()
    return {"success": False, "error": str(e)}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)


@app.websocket("/ws/progress_status")
async def websocket_progress_status(websocket: WebSocket):
  """WebSocket endpoint for progress prediction training status updates."""
  await websocket.accept()

  try:
    while True:
      if progress_trainer is None:
        await websocket.send_json(
            {"connected": False, "error": "Not connected to server"}
        )
        await asyncio.sleep(1)
        continue

      try:
        status = await asyncio.to_thread(
            progress_trainer.get_training_status  # type: ignore
        )

        await websocket.send_json(
            {
                "connected": True,
                "phase": status.phase,
                "is_finished": status.is_finished,
                "steps_completed": status.steps_completed,
                "max_steps": status.max_steps,
                "loss": status.loss if status.loss is not None else None,
                "accuracy": status.accuracy
                if status.accuracy is not None
                else None,
                "f1": status.f1 if status.f1 is not None else None,
                "fps": status.fps if status.fps is not None else None,
                "val_loss": status.val_loss
                if status.val_loss is not None
                else None,
                "val_accuracy": status.val_accuracy
                if status.val_accuracy is not None
                else None,
                "val_f1": status.val_f1 if status.val_f1 is not None else None,
                "checkpoint_id": status.checkpoint_id
                if status.checkpoint_id is not None
                else None,
                "export_entries_processed": status.export_entries_processed,
                "export_entries_total": status.export_entries_total,
                # Config for UI auto-fill on reconnect
                "model_name": status.model_name,
                "entry_filters": status.entry_filters,
                "batch_size": status.batch_size,
                "task_type": status.task_type,
            }
        )
      except Exception as e:
        await websocket.send_json({"connected": False, "error": str(e)})

      await asyncio.sleep(1)
  except WebSocketDisconnect:
    pass
