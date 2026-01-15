"""High-level SDK for robot control.

Re-exports client classes, futures, and RPC API data models.
"""

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
