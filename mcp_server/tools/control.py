"""Robot control tools: mode management, gripper, and movement."""

import json
from typing import Sequence

from mcp.server.fastmcp import Context

from r2_labs.mcp_server import serialization
from r2_labs.mcp_server import server
from r2_labs.sdk import rpc_api


@server.mcp.tool()
async def activate(ctx: Context) -> str:
  """Set the robot to READY mode for accepting behaviour commands."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.activate)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def deactivate(ctx: Context) -> str:
  """Set the robot to STOP mode, parking the arm at zero position."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.deactivate)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_execution_mode(ctx: Context) -> str:
  """Get the current robot execution mode."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.exec_mode.get_execution_mode)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def set_execution_mode(mode: str, ctx: Context) -> str:
  """Set the robot execution mode.

  Args:
    mode: one of STOP, READY, TEACH, TELEOP, DATA_COLLECTION_TELEOP.
  """
  robot = server.get_robot(ctx)
  execution_mode = server.parse_enum(rpc_api.ExecutionMode, mode)
  result = await server.run_sdk(
      ctx, robot.exec_mode.set_execution_mode, execution_mode
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def open_gripper(
    ctx: Context,
    target_position: float | None = None,
    timeout_seconds: float | None = None,
) -> str:
  """Open the gripper.

  Args:
    target_position: target gripper position, or None for fully open.
    timeout_seconds: max seconds to wait for completion.
  """
  robot = server.get_robot(ctx)
  future = await server.run_sdk(
      ctx,
      robot.arm.open_gripper,
      target_position=target_position,
      timeout=timeout_seconds,
  )
  result = await server.run_sdk(ctx, future.result)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def close_gripper(
    ctx: Context,
    target_position: float | None = None,
    timeout_seconds: float | None = None,
) -> str:
  """Close the gripper.

  Args:
    target_position: target gripper position, or None for fully closed.
    timeout_seconds: max seconds to wait for completion.
  """
  robot = server.get_robot(ctx)
  future = await server.run_sdk(
      ctx,
      robot.arm.close_gripper,
      target_position=target_position,
      timeout=timeout_seconds,
  )
  result = await server.run_sdk(ctx, future.result)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def go_to_neutral_pose(
    ctx: Context,
    timeout_seconds: float | None = None,
) -> str:
  """Move the arm to its neutral pose."""
  robot = server.get_robot(ctx)
  future = await server.run_sdk(
      ctx, robot.arm.go_to_neutral_pose, timeout=timeout_seconds
  )
  result = await server.run_sdk(ctx, future.result)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def go_to_joints(
    joint_positions: Sequence[float],
    ctx: Context,
    timeout_seconds: float | None = None,
) -> str:
  """Move the arm to specific joint positions.

  Args:
    joint_positions: target joint angles in radians.
    timeout_seconds: max seconds to wait for completion.
  """
  robot = server.get_robot(ctx)
  arm = server.get_arm(ctx)
  future = await server.run_sdk(
      ctx,
      robot.behaviour.go_to_joints,
      configuration=joint_positions,
      timeout=timeout_seconds,
      arm=arm,
  )
  result = await server.run_sdk(ctx, future.result)
  return json.dumps(serialization.serialize(result))
