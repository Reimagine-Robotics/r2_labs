"""FastMCP server for R2 robot control."""

import argparse
import asyncio
import enum
import functools
import os
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import futures as sdk_futures
from r2_labs.sdk import rpc_api

_T = TypeVar("_T")


def parse_enum(enum_cls: type[enum.Enum], value: str) -> Any:
  """Parse a string into an enum value with a clear error message."""
  try:
    return enum_cls[value.upper()]
  except KeyError:
    valid = ", ".join(m.name for m in enum_cls)
    raise ValueError(
        f"invalid value {value!r}, expected one of: {valid}"
    ) from None


def _parse_arm(value: str) -> sdk_futures.ArmSide:
  """Parse an arm selection string into an ArmSide enum."""
  mapping = {
      "left": sdk_futures.ArmSide.LEFT,
      "right": sdk_futures.ArmSide.RIGHT,
  }
  normalized = value.strip().lower()
  if normalized not in mapping:
    raise ValueError(
        f"invalid arm selection: {value!r}, expected 'left' or 'right'"
    )
  return mapping[normalized]


@asynccontextmanager
async def _robot_lifespan(
    server: FastMCP,  # pyright: ignore[reportUnusedParameter]
) -> AsyncIterator[dict[str, Any]]:
  """Manage Robot and executor lifecycle."""
  host = os.environ.get("R2_SERVER_HOST", "localhost")
  arm = _parse_arm(os.environ.get("R2_PRIMARY_ARM", "left"))

  robot = sdk_client.Robot(
      server_address=f"tcp://{host}:{rpc_api.DEFAULT_PORT}",
      query_server_address=f"tcp://{host}:{rpc_api.DEFAULT_QUERY_PORT}",
      training_server_address=f"tcp://{host}:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
      primary_arm=arm,
  )
  # single worker: SDK uses thread-local ZMQ sockets and the robot
  # serializes behaviour execution, so concurrent calls aren't useful.
  executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-sdk")
  try:
    yield {"robot": robot, "executor": executor, "arm": arm}
  finally:
    executor.shutdown(wait=True, cancel_futures=True)


def _build_instructions() -> str:
  """Build server instructions based on the runtime environment."""
  lines = [
      "This server provides tools for live robot control and resources with"
      " SDK source code. The tools and the Python SDK are separate APIs.",
      "",
      "When writing Python scripts that use the SDK:",
      "- Read the robot://sdk/client resource first for the actual API",
      "- The SDK uses robot.raw_robot.get_camera_data(), not get_camera_image()",
      "- Import as: from r2_labs.sdk import client, rpc_api",
      "- See robot://sdk/examples/{name} for usage patterns",
      "",
      "Do NOT use MCP tool names as SDK method names — they are different"
      " interfaces.",
  ]
  if os.environ.get("R2_IDE_BRIDGE_PORT"):
    lines += [
        "",
        "A UI is connected (VS Code extension). You have navigation tools:",
        "- open_ui_page: open library/behaviour/collect-data pages",
        "- prefill_add_object, prefill_add_trajectory,"
        " prefill_add_visual_pose, prefill_add_visual_trajectory:"
        " open add wizards with pre-filled fields",
        "",
        "For workflows that require human interaction (annotation, recording,"
        " arm movement), use these tools to guide the user to the right UI"
        " page. For example, to help add an object: prefill the add object"
        " wizard with the name, then instruct the user to annotate in the UI.",
    ]
  return "\n".join(lines)


# singleton server — tools register on this at import time.
# dns rebinding protection is disabled because the HTTP transport is intended
# for trusted local-network use alongside the robot's own RPC servers.
mcp = FastMCP(
    "r2-robot",
    instructions=_build_instructions(),
    lifespan=_robot_lifespan,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    ),
)


async def run_sdk(
    ctx: Context, fn: Callable[..., _T], *args: Any, **kwargs: Any
) -> _T:
  """Run a blocking SDK call in the executor thread."""
  executor = ctx.request_context.lifespan_context["executor"]  # type: ignore[union-attr]
  loop = asyncio.get_running_loop()
  return await loop.run_in_executor(
      executor, functools.partial(fn, *args, **kwargs)
  )


def get_robot(ctx: Context) -> sdk_client.Robot:
  """Get the Robot instance from the lifespan context."""
  return ctx.request_context.lifespan_context["robot"]  # type: ignore[union-attr]


def get_arm(ctx: Context) -> sdk_futures.ArmSide:
  """Get the configured primary arm from the lifespan context."""
  return ctx.request_context.lifespan_context["arm"]  # type: ignore[union-attr]


# register all tool modules
from r2_labs.mcp_server.tools import register_all  # noqa: E402

register_all(mcp)


def main() -> None:
  """CLI entry point for the MCP server."""
  parser = argparse.ArgumentParser(description="R2 Robot MCP Server")
  parser.add_argument(
      "--transport",
      choices=["stdio", "http"],
      default="stdio",
      help="transport protocol (default: stdio)",
  )
  parser.add_argument(
      "--port",
      type=int,
      default=8080,
      help="HTTP port (only used with --transport http)",
  )
  args = parser.parse_args()

  if args.transport == "http":
    import uvicorn

    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
  else:
    mcp.run(transport="stdio")
