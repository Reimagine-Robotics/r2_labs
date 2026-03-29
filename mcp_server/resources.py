"""MCP resources serving SDK source files for agent context."""

from pathlib import Path

from r2_labs.mcp_server import server

_SDK_DIR = Path(__file__).resolve().parent.parent / "sdk"
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "scripts"


@server.mcp.resource("robot://sdk/client")
def sdk_client() -> str:
  """R2 Labs SDK client source — Robot class and all sub-clients."""
  return (_SDK_DIR / "client.py").read_text()


@server.mcp.resource("robot://sdk/rpc-api")
def sdk_rpc_api() -> str:
  """R2 Labs SDK RPC API — all dataclass models and enums."""
  return (_SDK_DIR / "rpc_api.py").read_text()


@server.mcp.resource("robot://sdk/futures")
def sdk_futures() -> str:
  """R2 Labs SDK futures — CancellableFuture and ArmSelection."""
  return (_SDK_DIR / "futures.py").read_text()


@server.mcp.resource("robot://sdk/examples/{name}")
def sdk_example(name: str) -> str:
  """R2 Labs SDK example script."""
  resolved = (_EXAMPLES_DIR / name).resolve()
  if not resolved.is_relative_to(_EXAMPLES_DIR.resolve()):
    raise ValueError(f"path escapes examples directory: {name!r}")
  if not resolved.is_file():
    available = [f.name for f in _EXAMPLES_DIR.glob("*.py")]
    raise FileNotFoundError(
        f"example {name!r} not found. available: {available}"
    )
  return resolved.read_text()
