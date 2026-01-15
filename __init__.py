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
    ArmClient,
    ExecModeClient,
    ObjectLibraryClient,
    RawRobotClient,
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

__all__ = [
    "client",
    "futures",
    "rpc_api",
    "ArmClient",
    "BehaviourClient",
    "ExecModeClient",
    "ObjectLibraryClient",
    "RawRobotClient",
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

if hasattr(rpc_api, "__all__"):
  __all__ += list(rpc_api.__all__)
