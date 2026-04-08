"""Tool registration for the R2 MCP server."""

from mcp.server.fastmcp import FastMCP


def register_all(server: FastMCP) -> None:
  """Import all tool modules to register their tools with the server."""
  from r2_labs.mcp_server import resources
  from r2_labs.mcp_server.tools import (
      behaviour,
      control,
      library,
      navigation,
      recording,
      status,
      trainer,
  )

  del resources, behaviour, control, library, navigation, recording, status
  del trainer
