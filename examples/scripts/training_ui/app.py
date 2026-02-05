"""R2 Training Studio - FastAPI Backend."""

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bot01.data_warehouse import metadata_store

if TYPE_CHECKING:
  from r2_labs.sdk import client as sdk_client

app = FastAPI(title="R2 Training UI")

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global trainer clients (set via /connect endpoint)
trainer: "sdk_client.TrainerClient | None" = None  # Skill model trainer
progress_trainer: "sdk_client.ProgressPredictionTrainerClient | None" = None  # Progress prediction trainer
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


@app.get("/")
async def root():
  """Serve the main UI."""
  index_path = static_dir / "index.html"
  return FileResponse(index_path)


@app.get("/api/server_info")
async def server_info():
  """Get info about the UI server itself."""
  import socket

  hostname = socket.gethostname()
  return {
      "hostname": hostname,
      "port": 8000,
  }


@app.post("/api/connect")
async def connect(request: ConnectRequest):
  """Connect to the training server."""
  global trainer, progress_trainer, server_address

  # Lazy import to avoid loading heavy dependencies at module load time
  from r2_labs.rpc import client as rpc_client
  from r2_labs.sdk import client as sdk_client

  try:
    server_addr = f"tcp://{request.host}:{request.port}"

    # Create base client
    base_client = rpc_client.BaseClient(server_addr, timeout=5000)

    # Create both trainer clients
    test_trainer = sdk_client.TrainerClient(base_client)
    test_progress_trainer = sdk_client.ProgressPredictionTrainerClient(base_client)

    # Test connection with actual RPC call
    status = test_trainer.get_training_status()

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

  # Lazy import
  from r2_labs.rpc import client as rpc_client
  from r2_labs.sdk import client as sdk_client

  try:
    # Store server address before destroying trainer
    server_addr = server_address

    # Reset both trainers on server side (cancels training and clears state)
    print("[Hard Reset] Resetting trainers on server...")

    # Reset flow matching trainer
    try:
      if trainer:
        reset_response = trainer.reset_trainer()  # type: ignore
        if not reset_response.success:
          print(f"[Hard Reset] Flow matching reset failed: {reset_response.error}")
        else:
          print("[Hard Reset] Flow matching trainer reset successful")
    except Exception as e:
      print(f"[Hard Reset] Flow matching reset failed: {e}")

    # Reset progress prediction trainer
    try:
      if progress_trainer:
        reset_response = progress_trainer.reset_trainer()  # type: ignore
        if not reset_response.success:
          print(f"[Hard Reset] Progress trainer reset failed: {reset_response.error}")
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
    base_client = rpc_client.BaseClient(server_addr, timeout=5000)
    new_trainer = sdk_client.TrainerClient(base_client)
    new_progress_trainer = sdk_client.ProgressPredictionTrainerClient(base_client)

    # Test connections
    status = new_trainer.get_training_status()
    progress_status = new_progress_trainer.get_training_status()

    # Set new trainers
    trainer = new_trainer
    progress_trainer = new_progress_trainer

    print(f"[Hard Reset] Success - skill phase={status.phase}, progress phase={progress_status.phase}")

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
    status = trainer.get_training_status()  # type: ignore
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
    models = trainer.list_models()  # type: ignore
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
    all_names = trainer.list_model_names_from_checkpoints()  # type: ignore
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
  """Get available entry filter IDs from the data warehouse.

  Entry IDs format: entry_filter_id#hash
  Returns only the unique entry_filter_id portions.

  Args:
      search: Optional search term to filter results.

  Returns:
      List of unique entry_filter_ids matching the search term.
  """
  try:
    # Connect to metadata store
    api_cfg = metadata_store.ApiConfig(
        base_url="http://localhost:8081",
        auth_required=False,
    )
    store = metadata_store.ApiMetadataStore(api_cfg)
    reader = metadata_store.ApiMetadataReader(store)

    # Get all entry IDs
    search_pattern = f"{search}*" if search else "*"
    entry_ids = reader.get_entry_ids(entry_filter=search_pattern)

    # Extract unique entry_filter_ids (before the # hash)
    # Format: entry_filter_id#hash -> extract entry_filter_id
    filter_ids = set()
    for entry_id in entry_ids:
      if "#" in entry_id:
        filter_id = entry_id.split("#")[0]
        filter_ids.add(filter_id)
      else:
        # No hash - use the whole entry_id
        filter_ids.add(entry_id)

    # Filter to only show entries containing "rectify"
    rectify_filters = [f for f in filter_ids if "rectify" in f.lower()]

    # Return sorted list (limit to 100 for UI performance)
    results = sorted(list(rectify_filters))[:100]
    print(f"[Entry Filters] Showing {len(results)} rectify filters (filtered out {len(filter_ids) - len(rectify_filters)} non-rectify)")
    return {"success": True, "filters": results}

  except Exception as e:
    return {"success": False, "error": str(e), "filters": []}


@app.post("/api/train")
async def start_training(request: TrainRequest):
  """Start training."""
  if trainer is None:
    return {"success": False, "error": "Not connected to server"}

  try:
    response = trainer.train_skill_model(  # type: ignore
        model_name=request.model_name,
        training_steps=request.training_steps,
        entry_filters=request.entry_filters,
        batch_size=request.batch_size,
        prediction_horizon=request.prediction_horizon,
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
    response = trainer.cancel_training()  # type: ignore
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
    response = trainer.start_export(checkpoint_step=None)  # type: ignore
    if response.error:
      print(f"[Export] Start failed: {response.error}")
      return {"success": False, "error": response.error}

    print("[Export] Polling for completion...")
    # Poll for completion (max 60 seconds)
    for i in range(60):
      status = trainer.get_export_status()  # type: ignore
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

    response = progress_trainer.train_model(  # type: ignore
        model_name=request["model_name"],
        training_steps=request["training_steps"],
        entry_filters=entry_filters,
        human_entry_filters=human_entry_filters,
        batch_size=request.get("batch_size", 32),
        task_type=request.get("task_type", "classification"),
        cameras=cameras,
        force_rebuild=request.get("force_rebuild", False),
        checkpoint_interval_steps=request.get("checkpoint_interval_steps", 1000),
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
    response = progress_trainer.cancel_training()  # type: ignore
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
    status = progress_trainer.get_training_status()  # type: ignore
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
    return {"success": False, "error": "Not connected to server", "checkpoints": []}

  try:
    response = progress_trainer.list_checkpoints()  # type: ignore
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
    print(f"[Progress Export] Starting export from checkpoint: {checkpoint_step}")
    response = progress_trainer.start_export(checkpoint_step=checkpoint_step)  # type: ignore
    if response.error:
      print(f"[Progress Export] Start failed: {response.error}")
      return {"success": False, "error": response.error}

    print("[Progress Export] Polling for completion...")
    # Poll for completion (max 60 seconds)
    for i in range(60):
      status = progress_trainer.get_export_status()  # type: ignore
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
    response = progress_trainer.reset_trainer()  # type: ignore
    return {"success": response.success, "error": response.error}
  except Exception as e:
    print(f"[Progress Reset] Exception: {e}")
    traceback.print_exc()
    return {"success": False, "error": str(e)}


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
  """WebSocket endpoint for live training status updates."""
  await websocket.accept()

  try:
    while True:
      if trainer is None:
        await websocket.send_json(
            {"connected": False, "error": "Not connected to server"}
        )
        await asyncio.sleep(1)
        continue

      try:
        status = trainer.get_training_status()  # type: ignore

        # Debug logging
        print(
            f"[WebSocket] Phase: {status.phase}, Export: {status.export_entries_processed}/{status.export_entries_total}"
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
        await websocket.send_json({"connected": False, "error": str(e)})

      await asyncio.sleep(1)  # Poll every second

  except WebSocketDisconnect:
    pass


if __name__ == "__main__":
  import uvicorn

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
        status = progress_trainer.get_training_status()  # type: ignore

        await websocket.send_json(
            {
                "connected": True,
                "phase": status.phase,
                "is_finished": status.is_finished,
                "steps_completed": status.steps_completed,
                "max_steps": status.max_steps,
                "loss": status.loss if status.loss is not None else None,
                "accuracy": status.accuracy if status.accuracy is not None else None,
                "f1": status.f1 if status.f1 is not None else None,
                "fps": status.fps if status.fps is not None else None,
                "val_loss": status.val_loss if status.val_loss is not None else None,
                "val_accuracy": status.val_accuracy if status.val_accuracy is not None else None,
                "val_f1": status.val_f1 if status.val_f1 is not None else None,
                "checkpoint_id": status.checkpoint_id if status.checkpoint_id is not None else None,
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
