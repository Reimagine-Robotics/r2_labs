"""Tests that MCP behaviour tools call the SDK with arguments it accepts.

`server.run_sdk` forwards a tool's keyword arguments straight to the SDK method
via `functools.partial`, with no filtering — so a tool advertising an argument
the SDK does not take raises `TypeError` on *every* invocation, whether or not
the caller supplied it. Nothing else exercises these tools, so the mismatch is
invisible until someone drives the robot through MCP.

Autospeccing the client makes the mock enforce the real signatures, which turns
that class of mismatch into a test failure here.
"""

import asyncio
from unittest import mock

from r2_labs.mcp_server import server
from r2_labs.mcp_server.tools import behaviour
from r2_labs.sdk import client


async def _passthrough_run_sdk(ctx, fn, *args, **kwargs):
  """Bind and call inline, exactly as run_sdk does minus the executor."""
  del ctx
  return fn(*args, **kwargs)


def _invoke(tool, **tool_kwargs) -> None:
  """Run an MCP behaviour tool against a signature-enforcing SDK client."""
  arm = mock.create_autospec(client.ArmClient, instance=True)
  robot = mock.MagicMock()
  robot.arm = arm

  with (
      mock.patch.object(server, "get_robot", return_value=robot),
      mock.patch.object(server, "run_sdk", _passthrough_run_sdk),
  ):
    asyncio.run(tool(ctx=mock.MagicMock(), **tool_kwargs))


def test_execute_visual_trajectory_matches_the_sdk_signature():
  # Regression: this forwarded `period_seconds`, which
  # ArmClient.visual_trajectory_motion does not accept, so the tool failed on
  # every call. Visual trajectories have no period concept at all.
  _invoke(
      behaviour.execute_visual_trajectory, visual_trajectory_name="pick_drive"
  )


def test_execute_trajectory_matches_the_sdk_signature():
  # The sibling tool this was copy-pasted from. Plain trajectories do take a
  # period, so passing one here must remain valid - otherwise the test above
  # would pass for the wrong reason.
  _invoke(
      behaviour.execute_trajectory,
      trajectory_name="observe_infeed",
      period_seconds=2.0,
  )
