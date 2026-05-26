"""Behaviour execution, perception queries, and apriltag detection."""

import json
from typing import Sequence

from mcp.server.fastmcp import Context

from r2_labs.mcp_server import serialization, server
from r2_labs.sdk import rpc_api


async def _execute_behaviour(
    ctx: Context,
    future: object,
) -> str:
  """Await a behaviour future and serialize the result."""
  result = await server.run_sdk(ctx, future.result)  # type: ignore[union-attr]
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def execute_trajectory(
    trajectory_name: str,
    ctx: Context,
    timeout_seconds: float | None = 30.0,
    period_seconds: float | None = None,
    motion_type: str = "FULL",
    static_gripper: bool = False,
) -> str:
  """Execute a named trajectory from the trajectory library.

  Args:
    trajectory_name: name of the trajectory to execute.
    timeout_seconds: max seconds to wait for completion.
    period_seconds: duration override for execution.
    motion_type: FULL, GO_TO_START, or GO_TO_END.
    static_gripper: whether to keep the gripper static during motion.
  """
  robot = server.get_robot(ctx)
  mt = server.parse_enum(rpc_api.TrajectoryMotionType, motion_type)
  future = await server.run_sdk(
      ctx,
      robot.arm.trajectory_motion,
      trajectory_name=trajectory_name,
      timeout=timeout_seconds,
      period_seconds=period_seconds,
      motion_type=mt,
      static_gripper=static_gripper,
  )
  return await _execute_behaviour(ctx, future)


@server.mcp.tool()
async def execute_visual_trajectory(
    visual_trajectory_name: str,
    ctx: Context,
    timeout_seconds: float | None = 30.0,
    period_seconds: float | None = None,
    static_gripper: bool = False,
    motion_type: str = "FULL",
) -> str:
  """Execute a named visual trajectory.

  Args:
    visual_trajectory_name: name of the visual trajectory to execute.
    timeout_seconds: max seconds to wait for completion.
    period_seconds: duration override, or None to use recorded duration.
    static_gripper: whether to keep the gripper static during motion.
    motion_type: FULL, GO_TO_START, or GO_TO_END.
  """
  robot = server.get_robot(ctx)
  mt = server.parse_enum(rpc_api.TrajectoryMotionType, motion_type)
  future = await server.run_sdk(
      ctx,
      robot.arm.visual_trajectory_motion,
      visual_trajectory_name=visual_trajectory_name,
      period_seconds=period_seconds,
      timeout=timeout_seconds,
      static_gripper=static_gripper,
      motion_type=mt,
  )
  return await _execute_behaviour(ctx, future)


@server.mcp.tool()
async def execute_visual_pose_motion(
    visual_pose_name: str,
    period_seconds: float,
    ctx: Context,
    timeout_seconds: float | None = 30.0,
) -> str:
  """Execute a named visual pose motion.

  Args:
    visual_pose_name: name of the visual pose to execute.
    period_seconds: duration for the motion in seconds.
    timeout_seconds: max seconds to wait for completion.
  """
  robot = server.get_robot(ctx)
  future = await server.run_sdk(
      ctx,
      robot.arm.visual_pose_motion,
      visual_pose_name=visual_pose_name,
      period_seconds=period_seconds,
      timeout=timeout_seconds,
  )
  return await _execute_behaviour(ctx, future)


@server.mcp.tool()
async def execute_learned_behavior(
    model_id: str,
    ctx: Context,
    timeout_seconds: float | None = None,
    prefer_service: bool = True,
    task: str = "",
) -> str:
  """Execute a learned behaviour model.

  Args:
    model_id: ID of the trained model to execute.
    timeout_seconds: max seconds to wait for completion.
    prefer_service: if True, prefer a running inference service over local inference.
    task: per-step language instruction for lerobot VLA policies. Empty
      falls back to the server's --cfg.default_task and is ignored when the
      server reports wire_format='bc'.
  """
  robot = server.get_robot(ctx)
  arm = server.get_arm(ctx)
  query = rpc_api.ExecuteLearnedBehaviorQuery(
      model_id=model_id,
      timeout_seconds=timeout_seconds,
      prefer_service=prefer_service,
      task=task,
  )
  future = await server.run_sdk(
      ctx,
      robot.behaviour.execute_learned_behavior,
      query=query,
      timeout=timeout_seconds,
      arm=arm,
  )
  return await _execute_behaviour(ctx, future)


@server.mcp.tool()
async def can_see_object(
    object_names: Sequence[str],
    ctx: Context,
    timeout_seconds: float = 30.0,
) -> str:
  """Check if any of the specified objects are visible to the robot.

  Args:
    object_names: names of objects to look for.
    timeout_seconds: max seconds to wait for detection.
  """
  robot = server.get_robot(ctx)
  result = await server.run_sdk(
      ctx,
      robot.query.can_see_object,
      object_names=object_names,
      timeout_seconds=timeout_seconds,
  )
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def detect_apriltags(
    ctx: Context,
    camera: str = "WRIST",
    tag_size: float | None = None,
) -> str:
  """Detect AprilTags in a camera image.

  Args:
    camera: camera to use — WRIST, SCENE_LEFT, or SCENE_RIGHT.
    tag_size: physical tag size in meters, or None for default.
  """
  robot = server.get_robot(ctx)
  camera_type = server.parse_enum(rpc_api.CameraType, camera)
  result = await server.run_sdk(
      ctx,
      robot.detect_apriltags_from_camera,
      camera=camera_type,
      tag_size=tag_size,
  )
  return json.dumps(serialization.serialize(result))
