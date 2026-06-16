"""Navigation tools for directing users to UI pages."""

import asyncio
import json
import os
import urllib.request

from mcp.server.fastmcp import Context

from r2_labs.mcp_server import server

# keep in sync with Category type in bot01/hri/ide/extension/src/webview/types/library.ts
_VALID_CATEGORIES = frozenset(
    {
        "behaviour",
        "object",
        "trajectory",
        "visualPose",
        "visualTrajectory",
        "collectData",
    }
)


def _get_bridge_url() -> str | None:
  """Get the UI bridge base URL from environment, or None if unavailable."""
  port = os.environ.get("R2_IDE_BRIDGE_PORT")
  if not port:
    return None
  return f"http://127.0.0.1:{port}"


def _post_to_bridge(path: str, payload: dict) -> str:
  """POST JSON to the UI bridge and return the response body."""
  bridge_url = _get_bridge_url()
  if bridge_url is None:
    return json.dumps(
        {
            "error": "no_ui",
            "message": (
                "UI navigation requires a connected UI"
                " (VS Code extension or webapp)."
            ),
        }
    )
  data = json.dumps(payload).encode()
  req = urllib.request.Request(
      f"{bridge_url}{path}",
      data=data,
      headers={"Content-Type": "application/json"},
      method="POST",
  )
  try:
    with urllib.request.urlopen(req, timeout=5) as resp:
      return resp.read().decode()
  except Exception as e:
    return json.dumps({"error": "bridge_error", "message": str(e)})


async def _bridge_call(path: str, payload: dict) -> str:
  """POST to the UI bridge without blocking the SDK executor."""
  return await asyncio.to_thread(_post_to_bridge, path, payload)


@server.mcp.tool()
async def open_ui_page(
    category: str,
    ctx: Context,  # pyright: ignore[reportUnusedParameter]
    item: str | None = None,
) -> str:
  """Open a page in the Reimagine Robotics UI.

  Args:
    category: page to open — behaviour, object, trajectory,
              visualPose, visualTrajectory, or collectData.
    item: optional item name to select within the page.
  """
  if category not in _VALID_CATEGORIES:
    valid = ", ".join(sorted(_VALID_CATEGORIES))
    raise ValueError(f"invalid category {category!r}, expected one of: {valid}")
  payload: dict = {"category": category}
  if item is not None:
    payload["item"] = item
  return await _bridge_call("/navigate", payload)


async def _prefill(wizard: str, data: dict) -> str:
  """Send a prefill-wizard request to the UI bridge."""
  return await _bridge_call("/prefill-wizard", {"wizard": wizard, "data": data})


@server.mcp.tool()
async def prefill_add_object(
    ctx: Context,  # pyright: ignore[reportUnusedParameter]
    name: str | None = None,
    description: str | None = None,
) -> str:
  """Open the Add Object wizard in the UI with pre-filled fields.

  Args:
    name: pre-fill the object name.
    description: pre-fill the object description.
  """
  data: dict = {}
  if name is not None:
    data["name"] = name
  if description is not None:
    data["description"] = description
  return await _prefill("addObject", data)


@server.mcp.tool()
async def prefill_add_trajectory(
    ctx: Context,  # pyright: ignore[reportUnusedParameter]
    name: str | None = None,
    description: str | None = None,
    trajectory_type: str | None = None,
    trajectory_source: str | None = None,
) -> str:
  """Open the Add Trajectory wizard in the UI with pre-filled fields.

  Args:
    name: pre-fill the trajectory name.
    description: pre-fill the trajectory description.
    trajectory_type: JOINT_ABSOLUTE or WRIST_CARTESIAN_RELATIVE.
    trajectory_source: ROBOT or TELEOP.
  """
  data: dict = {}
  if name is not None:
    data["name"] = name
  if description is not None:
    data["description"] = description
  if trajectory_type is not None:
    data["trajectoryType"] = trajectory_type
  if trajectory_source is not None:
    data["trajectorySource"] = trajectory_source
  return await _prefill("addTrajectory", data)


@server.mcp.tool()
async def prefill_add_visual_pose(
    ctx: Context,  # pyright: ignore[reportUnusedParameter]
    name: str | None = None,
    description: str | None = None,
) -> str:
  """Open the Add Visual Pose wizard in the UI with pre-filled fields.

  Args:
    name: pre-fill the visual pose name.
    description: pre-fill the visual pose description.
  """
  data: dict = {}
  if name is not None:
    data["name"] = name
  if description is not None:
    data["description"] = description
  return await _prefill("addVisualPose", data)


@server.mcp.tool()
async def prefill_add_visual_trajectory(
    ctx: Context,  # pyright: ignore[reportUnusedParameter]
    name: str | None = None,
    description: str | None = None,
    trajectory_source: str | None = None,
    hold_until_start: bool | None = None,
) -> str:
  """Open the Add Visual Trajectory wizard in the UI with pre-filled fields.

  Args:
    name: pre-fill the visual trajectory name.
    description: pre-fill the visual trajectory description.
    trajectory_source: ROBOT or TELEOP.
    hold_until_start: if True, hold position until recording starts.
  """
  data: dict = {}
  if name is not None:
    data["name"] = name
  if description is not None:
    data["description"] = description
  if trajectory_source is not None:
    data["trajectorySource"] = trajectory_source
  if hold_until_start is not None:
    data["holdUntilStart"] = hold_until_start
  return await _prefill("addVisualTrajectory", data)
