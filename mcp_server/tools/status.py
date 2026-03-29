"""Observation and health status tools."""

import json

from mcp.server.fastmcp import Context
from mcp.types import ImageContent, TextContent

from r2_labs.mcp_server import serialization
from r2_labs.mcp_server import server
from r2_labs.sdk import rpc_api


@server.mcp.tool()
async def get_arm_state(ctx: Context) -> str:
  """Get current arm proprioception: joint positions, velocities, efforts, and wrist pose."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.raw_robot.get_proprio_data)
  return json.dumps(serialization.serialize(result))


@server.mcp.tool()
async def get_camera_image(
    ctx: Context,
    camera: str = "WRIST",
) -> list[TextContent | ImageContent]:
  """Capture an image from a robot camera.

  Args:
    camera: camera to use — WRIST, SCENE_LEFT, or SCENE_RIGHT.
  """
  robot = server.get_robot(ctx)
  camera_type = server.parse_enum(rpc_api.CameraType, camera)
  result = await server.run_sdk(
      ctx, robot.raw_robot.get_camera_data, camera_type
  )

  content: list[TextContent | ImageContent] = []

  if result.rgb is not None:
    image_data = serialization.encode_image(result.rgb)
    content.append(
        ImageContent(type="image", data=image_data, mimeType="image/jpeg")
    )

  metadata = {
      "availability": serialization.serialize(result.availability),
      "has_rgb": result.rgb is not None,
      "has_depth": result.depth is not None,
      "rgb_shape": list(result.rgb.shape) if result.rgb is not None else None,
      "depth_shape": list(result.depth.shape)
      if result.depth is not None
      else None,
      "intrinsics": serialization.serialize(result.intrinsics)
      if result.intrinsics is not None
      else None,
  }
  content.append(TextContent(type="text", text=json.dumps(metadata)))

  return content


@server.mcp.tool()
async def get_hardware_health(ctx: Context) -> str:
  """Get hardware health status for all robot components."""
  robot = server.get_robot(ctx)
  result = await server.run_sdk(ctx, robot.hardware_health.get_status)
  return json.dumps(serialization.serialize(result))
