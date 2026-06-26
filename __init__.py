"""R2 Labs SDK for robot control and behaviour execution.

This package provides:
- RPC client/server infrastructure for robot communication
- High-level client classes for robot control (Robot, BehaviourClient, etc.)
- Async execution primitives (ArmExecutor, Future)
- Data models for queries and responses (rpc_api)
"""

import importlib.metadata
import pathlib
import tomllib

from r2_labs.sdk import client, futures, rpc_api  # noqa: F401
from r2_labs.sdk.client import BehaviourClient  # noqa: F401
from r2_labs.sdk.client import (
    AprilTagCameraDetection,
    AprilTagClient,
    ArmClient,
    BehaviourCancelledError,
    BehaviourFailedError,
    CollectDataClient,
    EpisodeObserverClient,
    ExecModeClient,
    ObjectAnnotationPoint,
    ObjectLibraryClient,
    QueryClient,
    RawRobotClient,
    RecordingClient,
    Robot,
    TrajectoryLibraryClient,
    VisualPoseLibraryClient,
)
from r2_labs.sdk.futures import FIRST_COMPLETED  # noqa: F401
from r2_labs.sdk.futures import (
    ALL_COMPLETED,
    FIRST_EXCEPTION,
    ArmExecutor,
    ArmSelection,
    ArmSide,
    Future,
    SingleThreadExecutor,
    as_completed,
    wait,
)
from r2_labs.sdk.rpc_api import *  # noqa: F401,F403


def get_version() -> str | None:
  """Resolved distribution version, or None if genuinely undeterminable.

  A pip-installed client wheel resolves via importlib.metadata. The backend
  runs r2_labs from the monorepo checkout (not installed under its own dist
  name), so it falls back to reading the checked-out pyproject.toml — which the
  release stamps to the release version. None when neither is available; callers
  that compare versions treat None as "unknown, don't compare".
  """
  try:
    return importlib.metadata.version("r2-labs")
  except importlib.metadata.PackageNotFoundError:
    pass
  pyproject = pathlib.Path(__file__).parent / "pyproject.toml"
  try:
    return tomllib.loads(pyproject.read_text())["project"]["version"]
  except (OSError, KeyError, tomllib.TOMLDecodeError):
    return None


# __version__ stays a valid version string by convention (PEP 396 is withdrawn,
# but consumers parse/format it as a str); "0.0.0" is the conventional unknown
# fallback. The API reports the un-floored get_version() (None when unknown).
__version__ = get_version() or "0.0.0"

__all__: list[str] = [
    "__version__",
    "get_version",
    "client",
    "futures",
    "rpc_api",
    "AprilTagCameraDetection",
    "AprilTagClient",
    "ArmClient",
    "BehaviourCancelledError",
    "BehaviourClient",
    "BehaviourFailedError",
    "CollectDataClient",
    "EpisodeObserverClient",
    "ExecModeClient",
    "ObjectAnnotationPoint",
    "ObjectLibraryClient",
    "QueryClient",
    "RawRobotClient",
    "RecordingClient",
    "Robot",
    "TrajectoryLibraryClient",
    "VisualPoseLibraryClient",
    "ALL_COMPLETED",
    "FIRST_COMPLETED",
    "FIRST_EXCEPTION",
    "ArmExecutor",
    "ArmSelection",
    "ArmSide",
    "Future",
    "SingleThreadExecutor",
    "as_completed",
    "wait",
]

_rpc_api_all = getattr(rpc_api, "__all__", None)
if isinstance(_rpc_api_all, (list, tuple)):
  __all__ += list(_rpc_api_all)
