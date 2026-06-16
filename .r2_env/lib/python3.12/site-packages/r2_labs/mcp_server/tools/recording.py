"""Trajectory and visual trajectory recording workflow tools."""

import json

from mcp.server.fastmcp import Context

from r2_labs.mcp_server import serialization
from r2_labs.mcp_server import server
from r2_labs.sdk import rpc_api

# --- trajectory recording ---


@server.mcp.tool()
async def prepare_recording(
    ctx: Context,
    trajectory_type: str = "JOINT_ABSOLUTE",
    trajectory_source: str = "ROBOT",
    timeout_seconds: float | None = 300.0,
    hold_until_start: bool = False,
) -> str:
  """Prepare a trajectory recording session.

  Args:
    trajectory_type: JOINT_ABSOLUTE, JOINT_RELATIVE, or WRIST_CARTESIAN_RELATIVE.
    trajectory_source: ROBOT or TELEOP.
    timeout_seconds: max seconds before the recording times out.
    hold_until_start: if True, hold position until start() is called.
  """
  robot = server.get_robot(ctx)
  tt = server.parse_enum(rpc_api.TrajectoryType, trajectory_type)
  ts = server.parse_enum(rpc_api.TrajectorySource, trajectory_source)
  result = await server.run_sdk(
      ctx,
      robot.recording.prepare,
      trajectory_type=tt,
      trajectory_source=ts,
      timeout_seconds=timeout_seconds,
      hold_until_start=hold_until_start,
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def start_recording(ctx: Context) -> str:
  """Start a previously prepared trajectory recording."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.recording.start)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def stop_recording(ctx: Context) -> str:
  """Stop the current trajectory recording."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.recording.stop)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_recording_state(ctx: Context) -> str:
  """Get the current state of the trajectory recording."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.recording.get_state)
  return json.dumps(serialization.serialize(result))


# --- visual trajectory recording ---


@server.mcp.tool()
async def prepare_visual_recording(
    ctx: Context,
    trajectory_source: str = "ROBOT",
    timeout_seconds: float | None = 300.0,
    hold_until_start: bool = False,
) -> str:
  """Prepare a visual trajectory recording session.

  Args:
    trajectory_source: ROBOT or TELEOP.
    timeout_seconds: max seconds before the recording times out.
    hold_until_start: if True, hold position until start() is called.
  """
  robot = server.get_robot(ctx)
  ts = server.parse_enum(rpc_api.TrajectorySource, trajectory_source)
  result = await server.run_sdk(
      ctx,
      robot.visual_trajectory_recording.prepare,
      trajectory_source=ts,
      timeout_seconds=timeout_seconds,
      hold_until_start=hold_until_start,
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def start_visual_recording(ctx: Context) -> str:
  """Start a previously prepared visual trajectory recording."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.visual_trajectory_recording.start)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def stop_visual_recording(ctx: Context) -> str:
  """Stop the current visual trajectory recording."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.visual_trajectory_recording.stop)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_visual_recording_state(ctx: Context) -> str:
  """Get the current state of the visual trajectory recording."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.visual_trajectory_recording.get_state
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def save_visual_recording(
    name: str,
    ctx: Context,
    description: str = "",
    reference_type: str = "OBJECT",
    camera: str = "WRIST",
    allow_overwrite: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> str:
  """Save the current visual recording to the library.

  Args:
    name: name for the saved visual trajectory.
    description: optional description.
    reference_type: visual reference type — OBJECT, AR_MARKER, APRILTAG, or NONE.
    camera: camera used — WRIST, SCENE_LEFT, or SCENE_RIGHT.
    allow_overwrite: if True, overwrite an existing entry with the same name.
    start_frame: optional start frame for trimming.
    end_frame: optional end frame for trimming.
  """
  robot = server.get_robot(ctx)
  ref = server.parse_enum(rpc_api.VisualReference, reference_type)
  cam = server.parse_enum(rpc_api.CameraType, camera)
  result = await server.run_sdk(
      ctx,
      robot.visual_trajectory_recording.save,
      name=name,
      description=description,
      reference_type=ref,
      camera_type=cam,
      allow_overwrite=allow_overwrite,
      start_frame=start_frame,
      end_frame=end_frame,
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def load_visual_recording(name: str, ctx: Context) -> str:
  """Load a saved visual trajectory into the recording buffer for review.

  Args:
    name: name of the saved visual trajectory to load.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.visual_trajectory_recording.load_from_saved, name
  )
  return json.dumps(serialization.serialize(result))
