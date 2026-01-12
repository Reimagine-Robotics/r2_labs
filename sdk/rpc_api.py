import dataclasses
import enum

import numpy as np

DEFAULT_PORT = 7532
DEFAULT_QUERY_PORT = DEFAULT_PORT + 1


@enum.unique
class ExecutionMode(enum.Enum):
  # In STOP mode, the arm is parked at the zero position and relaxed. While in
  # STOP mode, all execution commands are ignored, but you can still read
  # sensor data.
  STOP = enum.auto()

  # In READY mode, the arm is ready to receive high level commands (eg: execute
  # behaviour, go to object, etc). In this mode, the arm is still and cannot
  # be moved around kinesthetically.
  READY = enum.auto()

  # In TEACH mode, the arm responds to kinesthetic teaching and cuff button
  # presses. All high level and raw commands are ignored in this mode, however
  TEACH = enum.auto()

  # In TELEOP mode, the arm can be controlled via teleoperation.
  TELEOP = enum.auto()


@dataclasses.dataclass
class ExecutionModeQuery:

  # If the query mode is None, then mode remains unchanged.
  new_mode: ExecutionMode | None = None


@dataclasses.dataclass
class ExecutionModeQueryResponse:
  current_mode: ExecutionMode


########################
# Robot Sensor queries #
########################


@enum.unique
class CameraType(enum.Enum):
  WRIST = enum.auto()
  SCENE_LEFT = enum.auto()
  SCENE_RIGHT = enum.auto()


@dataclasses.dataclass
class CameraQuery:
  camera: CameraType


@dataclasses.dataclass
class CameraQueryResponse:
  rgb: np.ndarray | None = None
  depth: np.ndarray | None = None
  intrinsics: np.ndarray | None = None


@dataclasses.dataclass
class ArmStateQueryResponse:
  joint_positions: np.ndarray | None = None
  joint_velocities: np.ndarray | None = None
  joint_efforts: np.ndarray | None = None

  gripper_positions: np.ndarray | None = None
  gripper_efforts: np.ndarray | None = None
  wrist_pose: np.ndarray | None = None


@dataclasses.dataclass
class CuffBottonsQueryResponse:

  # The pressed state of each of the cuff buttons. True indicates the button is
  # currently pressed.
  buttons_state: tuple[bool, ...] | None = None


##########################
# Object Library queries #
##########################


@dataclasses.dataclass
class ObjectLibraryEntry:
  name: str
  description: str

  # An RGB image and a binary mask over that image that shows some view of the
  # object.
  preview_image: np.ndarray
  preview_mask: np.ndarray


@dataclasses.dataclass
class ListObjectsResponse:

  objects: list[ObjectLibraryEntry]


@dataclasses.dataclass
class DeleteObjectQuery:
  object_name: str


@dataclasses.dataclass
class DeleteObjectQueryResponse:
  success: bool


@dataclasses.dataclass
class DetectObjectQuery:
  name: str


@dataclasses.dataclass
class ObjectDetectionEntry:

  object_name: str

  # Axis aligned bounding box (in world coordinates) of detected object surface
  # points.
  aabb_centre: np.ndarray  # 3-vector
  aabb_extents: np.ndarray  # 3-vector symmetric half-extents from centre.

  # Some heuristic value between 0 and 1 indicating confidence of detection.
  # Probably something proportional to the number of matches patches across all
  # 3 cameras?
  confidence: float


@dataclasses.dataclass
class DetectObjectQueryResponse:

  detected_instances: list[ObjectDetectionEntry]


@dataclasses.dataclass
class ObjectSegmentationQuery:

  # [T H W 3] shaped tensor of RGB frames. Dtype is uint8.
  frames: np.ndarray

  # List of 3D (Time, Y, X) tensors of points specified by the user as being on
  # the object. The dtype of each tensor is int32.
  positive_points: list[np.ndarray]

  # List of 3D (Time, Y, X) tensors of points specified by the user as being not
  # on the object. The dtype of each tensor is int32.
  negative_points: list[np.ndarray]


@dataclasses.dataclass
class ObjectSegmentationQueryResponse:
  # [T H W] shaped tensor of segmentation masks. Dtype is np.bool_.
  segmentation_mask: np.ndarray


@dataclasses.dataclass
class AddObjectViewsQuery:

  object_name: str

  # Description of the object. If this is an empty string, then it is ignored.
  object_description: str

  # [T H W 3] shaped tensor of RGB frames. Dtype is uint8.
  frames: np.ndarray

  # [T H W] shaped tensor of segmentation masks. Dtype is can be anything that
  # converts to 1.0 and 0.0 when cast to a np.float32.
  segmentation_mask: np.ndarray


@dataclasses.dataclass
class AddObjectViewsQueryResponse:
  """Empty response, maybe will have data in the future."""


##############################
# Trajectory Library queries #
##############################


@enum.unique
class TrajectoryType(enum.Enum):
  """Enum for the different trajectory types."""

  # Absolute trajectory is a series of raw arm and gripper angles.
  JOINT_ABSOLUTE = enum.auto()

  # Joint relative trajctory is a series of arm and gripper angle deltas
  # that are relative to the arm configuration at the start of execution of the
  # trajectory.
  JOINT_RELATIVE = enum.auto()

  # Wrist cartesian relative trajectory is a series of 6-dof wrist poses
  # relative to the wrist pose at the start of execution of the trajectory.
  # The poses are expressed as XYZ position and Quaternion orientation.
  WRIST_CARTESIAN_RELATIVE = enum.auto()


@enum.unique
class TrajectorySource(enum.Enum):
  """Source of joint data for trajectory recording (for metadata tracking)."""

  # Recorded from main robot via kinesthetic teaching.
  ROBOT = enum.auto()

  # Recorded from teleop device with mirroring to main robot.
  TELEOP = enum.auto()


@dataclasses.dataclass
class TrajectoryLibraryEntry:
  """Entry in the trajectory library.

  Includes all of the information needed to replay a trajectory.
  """

  name: str

  description: str

  trajectory_type: TrajectoryType

  # The number of seconds the trajectory spans from start to end.
  period_seconds: float

  # The initial joint configuration of the arm/gripper at the start of the
  # trajectory, during definition time. This is only used for visualisation
  # purposes.
  trajectory_init: np.ndarray

  # The actual trajectory, a series of robot arm/gripper configurations through
  # time. The shape of this tensor is [N, D] where N is the time dimension and D
  # is either 7 (for joint+gripper trajectory types) or 8 (for TCP+gripper
  # trajectory type).
  trajectory_data: np.ndarray

  # The source of the joint data used to record this trajectory.
  trajectory_source: TrajectorySource


@dataclasses.dataclass
class ListTrajectoriesResponse:

  trajectories: list[TrajectoryLibraryEntry]


@dataclasses.dataclass
class AddTrajectoryQuery:
  trajectory: TrajectoryLibraryEntry

  # Whether to allow overwriting an existing trajectory with the same name.
  allow_overwrite: bool = False


@dataclasses.dataclass
class AddTrajectoryQueryResponse:
  success: bool


@dataclasses.dataclass
class DeleteTrajectoryQuery:
  trajectory_name: str


@dataclasses.dataclass
class DeleteTrajectoryQueryResponse:
  success: bool


@dataclasses.dataclass
class LoadTrajectoryQuery:
  trajectory_name: str


@dataclasses.dataclass
class LoadTrajectoryQueryResponse:
  trajectory: TrajectoryLibraryEntry | None


########################
# Recording queries    #
########################


@dataclasses.dataclass
class PrepareRecordingQuery:
  """Query to prepare for trajectory recording."""

  trajectory_type: TrajectoryType = TrajectoryType.JOINT_ABSOLUTE

  trajectory_source: TrajectorySource = TrajectorySource.ROBOT
  timeout_seconds: float | None = (
      30.0  # Auto-stop after duration, None = no limit
  )


@dataclasses.dataclass
class PrepareRecordingResponse:
  """Response after preparing for recording."""

  error: str | None = None


@dataclasses.dataclass
class StartRecordingResponse:
  """Response after starting recording."""

  error: str | None = None


@dataclasses.dataclass
class StopRecordingResponse:
  """Response after stopping recording, contains the recorded trajectory."""

  trajectory: TrajectoryLibraryEntry | None = None
  error: str | None = None


@dataclasses.dataclass
class RecordingStateResponse:
  """Response containing the current recording state."""

  is_recording: bool
  sample_count: int = 0
  trajectory_type: TrajectoryType | None = None
  trajectory_source: TrajectorySource | None = None
  timeout_seconds: float | None = None
  elapsed_seconds: float = 0.0
  timed_out: bool = False


###############################
# Visual Pose Library queries #
###############################


@enum.unique
class VisualReference(enum.Enum):
  """Enum for the different visual reference types."""

  # The visual reference is an Aruco marker
  AR_MARKER = enum.auto()

  # The visual reference is an object or mart of an object. Matching is done
  # with the UFM (or similar) model.
  OBJECT = enum.auto()


@dataclasses.dataclass
class VisualPoseEntry:

  # Name of the visual pose, must be unique.
  name: str

  # Human readable description of pose. Largely optional.
  description: str

  # What kind of visual reference is used for this pose.
  reference_type: VisualReference

  # RGB image from the wrist camera taken during pose definition time.
  # Tensor shape is [H W 3] and dtype np.uint8.
  rgb_image: np.ndarray

  # Depth image from the wrist camera taken during pose definition time.
  # Tensor shape is [H W 1] and dtype np.int16.
  depth_image: np.ndarray

  # Mask for the above RGB and Depth images, defining where the visual reference
  # is the image. Tensor shape is [H W]
  reference_mask: np.ndarray


@dataclasses.dataclass
class ListVisualPosesResponse:

  poses: list[VisualPoseEntry]


@dataclasses.dataclass
class AddVisualPoseQuery:
  pose: VisualPoseEntry

  # Whether to allow overwriting an existing trajectory with the same name.
  allow_overwrite: bool = False


@dataclasses.dataclass
class AddVisualPoseQueryResponse:
  success: bool


@dataclasses.dataclass
class DeleteVisualPoseQuery:
  pose_name: str


@dataclasses.dataclass
class DeleteVisualPoseQueryResponse:
  success: bool


@dataclasses.dataclass
class LoadVisualPoseQuery:
  pose_name: str


@dataclasses.dataclass
class LoadVisualPoseQueryResponse:
  pose: VisualPoseEntry | None


@dataclasses.dataclass
class VisualReferenceSegmentationQuery:

  # [H W 3] shaped RGB image tensor. Dtype is uint8.
  frame: np.ndarray

  # (N, Y, X) tensor of points specified by the user as being on the visual pose
  # reference. The dtype of each tensor is int32.
  positive_points: np.ndarray

  # (N, Y, X) tensor of points specified by the user as being not on the visual
  # pose reference. The dtype of each tensor is int32.
  negative_points: np.ndarray


@dataclasses.dataclass
class VisualReferenceSegmentationQueryResponse:
  # [H W] shaped tensor of segmentation masks. Dtype is np.bool_.
  segmentation_mask: np.ndarray


######################
# Behaviour  queries #
######################


@enum.unique
class TicketStatus(enum.Enum):
  """Status of a behaviour execution ticket."""

  PENDING = enum.auto()
  RUNNING = enum.auto()
  COMPLETED = enum.auto()
  FAILED = enum.auto()


@dataclasses.dataclass
class LogEntry:
  """A single log entry associated with a ticket."""

  timestamp: float
  level: str
  message: str


@dataclasses.dataclass
class TicketInfo:
  """Information about a behaviour execution ticket."""

  ticket_id: str
  status: TicketStatus
  behaviour_type: str
  created_at: float
  started_at: float | None = None
  finished_at: float | None = None
  result_data: dict | None = None
  termination_reason: str | None = None
  error_message: str | None = None


@dataclasses.dataclass
class BehaviourInitiatedResponse:
  """Response when a behaviour is initiated, returns ticket ID for tracking."""

  ticket_id: str
  error: str | None = None


@dataclasses.dataclass
class TicketStatusQuery:
  """Query the status of a behaviour ticket."""

  ticket_id: str


@dataclasses.dataclass
class TicketStatusResponse:
  """Response containing ticket status information."""

  info: TicketInfo | None = None
  not_found: bool = False


@dataclasses.dataclass
class TicketLogsQuery:
  """Query logs for a specific ticket."""

  ticket_id: str
  since_index: int = 0


@dataclasses.dataclass
class TicketLogsResponse:
  """Response containing logs for a ticket."""

  logs: list[LogEntry]
  next_index: int


@dataclasses.dataclass
class CancelTicketQuery:
  """Request to cancel a behaviour ticket."""

  ticket_id: str


@dataclasses.dataclass
class CancelTicketResponse:
  """Result of a cancel ticket request."""

  success: bool
  error: str | None = None


@dataclasses.dataclass
class ListTicketsQuery:
  """Query to list all tickets."""

  pass


@dataclasses.dataclass
class ListTicketsResponse:
  """Response containing all tickets."""

  tickets: list[TicketInfo]


# Behaviour initiation queries


@enum.unique
class TrajectoryMotionType(enum.Enum):
  # Execute the full trajectory sequence.
  FULL = enum.auto()

  # Move the arm/gripper to the start configuration of the trajectory.
  GO_TO_START = enum.auto()

  # Move the arm/gripper to the end configuration of the trajectory.
  GO_TO_END = enum.auto()


@dataclasses.dataclass
class TrajectoryMotionQuery:
  """Execute a trajectory from the trajectory library."""

  trajectory_name: str
  period_seconds: float | None = None

  # How to execute the trajectory. This can be either the full trajectory,
  # or just the start or end configuration.
  motion_type: TrajectoryMotionType = TrajectoryMotionType.FULL

  # If this is set to True, then the gripper component of the trajectory is
  # ignored and the gripper position does not change through the trajectory.
  static_gripper: bool = False


@dataclasses.dataclass
class OpenGripperQuery:
  """Open the gripper to a target position."""

  target_position: float = 0.07  # default 7cm open


@dataclasses.dataclass
class CloseGripperQuery:
  """Close the gripper to a target position."""

  target_position: float = 0.0


@dataclasses.dataclass
class WaitForObjectQuery:
  """Wait until one of the specified objects is detected."""

  object_names: list[str]
  timeout_seconds: float | None = None


@dataclasses.dataclass
class GoToJointsQuery:
  """Moves the arm to the given joint configuration.

  If the configuration is 6-dim, only the arm is moved and the gripper remains
  at its current position. If it is 7-dim, then the 7th dim is assumed to
  correspond to the gripper, and both the arm and gripper are moved.
  """

  configuration: np.ndarray


@dataclasses.dataclass
class GoToNeutralPoseQuery:
  """Move the arm to a neutral pose."""

  pass


@dataclasses.dataclass
class CanSeeObjectQuery:
  """Check if any of the specified objects are visible."""

  object_names: list[str]
  timeout_seconds: float = 15.0


@dataclasses.dataclass
class CanSeeObjectResponse:
  """Response for can see object query."""

  visible: bool
  object_name: str | None = None
  object_position: ObjectDetectionEntry | None = None
  error: str | None = None


@dataclasses.dataclass
class ObjectHeatmapQuery:
  """Get live heatmap for object detection."""

  object_name: str


@dataclasses.dataclass
class ObjectHeatmapResponse:
  """Heatmap visualization as base64 PNG."""

  image: str  # base64 data URI
  error: str | None = None


#############################
# Visualisation queries     #
#############################


@dataclasses.dataclass
class VisualisationUrlResponse:
  """Response containing the Rerun viewer URL."""

  url: str | None = None
  error: str | None = None
