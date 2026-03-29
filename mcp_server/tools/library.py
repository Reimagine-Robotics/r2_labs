"""CRUD tools for object, trajectory, visual trajectory, and visual pose libraries."""

import json

from mcp.server.fastmcp import Context

from r2_labs.mcp_server import serialization, server

# --- objects ---


@server.mcp.tool()
async def list_objects(ctx: Context) -> str:
  """List all objects in the object library."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.object_library.list_entries)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def delete_object(object_name: str, ctx: Context) -> str:
  """Delete an object from the object library.

  Args:
    object_name: name of the object to delete.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.object_library.delete_entry, object_name
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_object_heatmap(
    object_name: str,
    ctx: Context,
) -> str:
  """Get the detection heatmap for a named object.

  Args:
    object_name: name of the object.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.object_library.get_heatmap, object_name
  )
  return json.dumps(serialization.serialize(result))


# --- trajectories ---


@server.mcp.tool()
async def list_trajectories(ctx: Context) -> str:
  """List all trajectories in the trajectory library."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.trajectory_library.list_entries)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def load_trajectory(trajectory_name: str, ctx: Context) -> str:
  """Load a trajectory from the library into the active buffer.

  Args:
    trajectory_name: name of the trajectory to load.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.trajectory_library.load_entry, trajectory_name
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def delete_trajectory(trajectory_name: str, ctx: Context) -> str:
  """Delete a trajectory from the library.

  Args:
    trajectory_name: name of the trajectory to delete.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.trajectory_library.delete_entry, trajectory_name
  )
  return json.dumps(serialization.serialize(result))


# --- visual trajectories ---


@server.mcp.tool()
async def list_visual_trajectories(ctx: Context) -> str:
  """List all visual trajectories in the library."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.visual_trajectory_library.list_entries
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def load_visual_trajectory(
    visual_trajectory_name: str, ctx: Context
) -> str:
  """Load a visual trajectory from the library.

  Args:
    visual_trajectory_name: name of the visual trajectory to load.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.visual_trajectory_library.load_entry, visual_trajectory_name
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def delete_visual_trajectory(
    visual_trajectory_name: str, ctx: Context
) -> str:
  """Delete a visual trajectory from the library.

  Args:
    visual_trajectory_name: name of the visual trajectory to delete.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx,
      robot.visual_trajectory_library.delete_entry,
      visual_trajectory_name,
  )
  return json.dumps(serialization.serialize(result))


# --- visual poses ---


@server.mcp.tool()
async def list_visual_poses(ctx: Context) -> str:
  """List all visual poses in the library."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.visual_pose_library.list_entries)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def load_visual_pose(visual_pose_name: str, ctx: Context) -> str:
  """Load a visual pose from the library.

  Args:
    visual_pose_name: name of the visual pose to load.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.visual_pose_library.load_entry, visual_pose_name
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def delete_visual_pose(visual_pose_name: str, ctx: Context) -> str:
  """Delete a visual pose from the library.

  Args:
    visual_pose_name: name of the visual pose to delete.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx, robot.visual_pose_library.delete_entry, visual_pose_name
  )
  return json.dumps(serialization.serialize(result))
