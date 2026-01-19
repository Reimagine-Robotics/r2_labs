"""R2 Labs SDK for robot control and behaviour execution.

This package provides:
- RPC client/server infrastructure for robot communication
- High-level client classes for robot control (Robot, BehaviourClient, etc.)
- Async execution primitives (ArmExecutor, Future)
- Data models for queries and responses (rpc_api)
"""

from r2_labs.sdk import client, futures, rpc_api  # noqa: F401
from r2_labs.sdk.client import BehaviourClient  # noqa: F401
from r2_labs.sdk.client import (
    AprilTagCameraDetection,
    AprilTagClient,
    ArmClient,
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

__all__: list[str] = [
    "client",
    "futures",
    "rpc_api",
    "AprilTagCameraDetection",
    "AprilTagClient",
    "ArmClient",
    "BehaviourClient",
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
