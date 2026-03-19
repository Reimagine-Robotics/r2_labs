"""RPC API data models for robot control queries and responses."""

import dataclasses
import enum

import numpy as np

DEFAULT_PORT = 7532
DEFAULT_QUERY_PORT = DEFAULT_PORT + 1
DEFAULT_MODEL_TRAINER_PORT = DEFAULT_PORT + 2


@enum.unique
class ExecutionMode(enum.Enum):
  """Robot arm execution mode controlling available operations."""

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

  # TODO(akhil): Decide which teleop to use and remove the other.
  # In DATA_COLLECTION_TELEOP mode, the arm can be controlled via teleoperation.
  DATA_COLLECTION_TELEOP = enum.auto()


@dataclasses.dataclass
class ExecutionModeQuery:
  """Query to get or set the robot execution mode.

  Attributes:
    new_mode: Target mode to transition to, or None to query current mode.
  """

  # If the query mode is None, then mode remains unchanged.
  new_mode: ExecutionMode | None = None


@dataclasses.dataclass
class ExecutionModeQueryResponse:
  """Response containing the current execution mode.

  Attributes:
    current_mode: The robot's current execution mode after the query.
  """

  current_mode: ExecutionMode


########################
# Robot Sensor queries #
########################


@enum.unique
class CameraType(enum.Enum):
  """Available camera sources on the robot."""

  WRIST = enum.auto()
  SCENE_LEFT = enum.auto()
  SCENE_RIGHT = enum.auto()


@enum.unique
class CameraAvailability(enum.Enum):
  """Availability status for a camera source in current config/runtime state."""

  PRESENT = enum.auto()
  NOT_PRESENT = enum.auto()
  TEMPORARILY_UNAVAILABLE = enum.auto()


@dataclasses.dataclass
class CameraQuery:
  """Query to retrieve camera data.

  Attributes:
    camera: Which camera to read from.
  """

  camera: CameraType


@dataclasses.dataclass
class CameraQueryResponse:
  """Camera data response.

  Attributes:
    availability: Camera availability classification.
    rgb: RGB image as [H, W, 3] uint8 array, or None if unavailable.
    depth: Depth image as [H, W] array, or None if unavailable.
    intrinsics: Camera intrinsic matrix as [3, 3] array, or None.
  """

  availability: CameraAvailability = CameraAvailability.NOT_PRESENT
  rgb: np.ndarray | None = None
  depth: np.ndarray | None = None
  intrinsics: np.ndarray | None = None


@dataclasses.dataclass
class ArmStateQueryResponse:
  """Proprioceptive state of the robot arm.

  Attributes:
    joint_positions: Joint angles in radians as [N] array.
    joint_velocities: Joint velocities as [N] array.
    joint_efforts: Joint torques/forces as [N] array.
    gripper_positions: Gripper finger positions.
    gripper_efforts: Gripper forces.
    wrist_pose: End-effector pose as [7] array (xyz + quaternion).
  """

  joint_positions: np.ndarray | None = None
  joint_velocities: np.ndarray | None = None
  joint_efforts: np.ndarray | None = None

  gripper_positions: np.ndarray | None = None
  gripper_efforts: np.ndarray | None = None
  wrist_pose: np.ndarray | None = None


@enum.unique
class ButtonPeripheralSource(enum.Enum):
  """Source device for a button peripheral input."""

  CUFF = enum.auto()
  PEDAL = enum.auto()


@dataclasses.dataclass
class ButtonPeripheralQuery:
  """Query for button peripheral states."""


@dataclasses.dataclass
class ButtonPeripheralQueryResponse:
  """Raw button states for each control source.

  Attributes:
    buttons_by_source: Mapping from source to button states.
      - For cuff, state order is [A, B, C, D].
      - For pedal, state order is [A, B, C].
      - Value is None when source is unavailable.
  """

  buttons_by_source: dict[ButtonPeripheralSource, list[bool] | None]


##########################
# Object Library queries #
##########################


@dataclasses.dataclass
class ObjectLibraryEntry:
  """An object stored in the object library.

  Attributes:
    name: Unique identifier for the object.
    description: Human-readable description.
    preview_image: RGB preview as [H, W, 3] uint8 array.
    preview_mask: Binary mask as [H, W] bool array.
  """

  name: str
  description: str

  # An RGB image and a binary mask over that image that shows some view of the
  # object.
  preview_image: np.ndarray
  preview_mask: np.ndarray


@dataclasses.dataclass
class ListObjectsResponse:
  """Response containing all objects in the library.

  Attributes:
    objects: List of object entries.
  """

  objects: list[ObjectLibraryEntry]


@dataclasses.dataclass
class DeleteObjectQuery:
  """Query to delete an object from the library.

  Attributes:
    object_name: Name of the object to delete.
  """

  object_name: str


@dataclasses.dataclass
class DeleteObjectQueryResponse:
  """Response after attempting to delete an object.

  Attributes:
    success: True if the object was deleted.
  """

  success: bool


@dataclasses.dataclass
class DetectObjectQuery:
  """Query to detect a specific object in the scene.

  Attributes:
    name: Name of the object to detect.
  """

  name: str


@dataclasses.dataclass
class ObjectDetectionEntry:
  """A detected object instance in the scene.

  Attributes:
    object_name: Name of the detected object.
    aabb_centre: Center of axis-aligned bounding box in world coords [3].
    aabb_extents: Half-extents of the bounding box [3].
    confidence: Detection confidence score in [0, 1].
  """

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
  """Response containing detected object instances.

  Attributes:
    detected_instances: List of detected objects with positions.
  """

  detected_instances: list[ObjectDetectionEntry]


@dataclasses.dataclass
class ObjectSegmentationQuery:
  """Query to segment an object from video frames using point prompts.

  Attributes:
    frames: RGB video frames as [T, H, W, 3] uint8 array.
    positive_points: Points on the object as list of [N, 3] int32 (T, Y, X).
    negative_points: Points not on the object as list of [N, 3] int32 (T, Y, X).
  """

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
  """Response containing segmentation masks for the queried object.

  Attributes:
    segmentation_mask: Binary masks as [T, H, W] bool array.
  """

  # [T H W] shaped tensor of segmentation masks. Dtype is np.bool_.
  segmentation_mask: np.ndarray


@dataclasses.dataclass
class AddObjectViewsQuery:
  """Query to add views of an object to the library.

  Attributes:
    object_name: Name of the object (creates new or updates existing).
    object_description: Human-readable description (ignored if empty).
    frames: RGB video frames as [T, H, W, 3] uint8 array.
    segmentation_mask: Object masks as [T, H, W] array (castable to float).
  """

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
  """Response containing all trajectories in the library.

  Attributes:
    trajectories: List of trajectory entries.
  """

  trajectories: list[TrajectoryLibraryEntry]


@dataclasses.dataclass
class AddTrajectoryQuery:
  """Query to add a trajectory to the library.

  Attributes:
    trajectory: The trajectory entry to add.
    allow_overwrite: If True, overwrite existing trajectory with same name.
  """

  trajectory: TrajectoryLibraryEntry

  # Whether to allow overwriting an existing trajectory with the same name.
  allow_overwrite: bool = False


@dataclasses.dataclass
class AddTrajectoryQueryResponse:
  """Response after attempting to add a trajectory.

  Attributes:
    success: True if the trajectory was added.
  """

  success: bool


@dataclasses.dataclass
class DeleteTrajectoryQuery:
  """Query to delete a trajectory from the library.

  Attributes:
    trajectory_name: Name of the trajectory to delete.
  """

  trajectory_name: str


@dataclasses.dataclass
class DeleteTrajectoryQueryResponse:
  """Response after attempting to delete a trajectory.

  Attributes:
    success: True if the trajectory was deleted.
  """

  success: bool


@dataclasses.dataclass
class LoadTrajectoryQuery:
  """Query to load a trajectory from the library.

  Attributes:
    trajectory_name: Name of the trajectory to load.
  """

  trajectory_name: str


@dataclasses.dataclass
class LoadTrajectoryQueryResponse:
  """Response containing the loaded trajectory.

  Attributes:
    trajectory: The trajectory entry, or None if not found.
  """

  trajectory: TrajectoryLibraryEntry | None


########################
# Recording queries    #
########################d


@dataclasses.dataclass
class PrepareRecordingQuery:
  """Query to prepare for trajectory recording."""

  trajectory_type: TrajectoryType = TrajectoryType.JOINT_ABSOLUTE

  trajectory_source: TrajectorySource = TrajectorySource.ROBOT
  timeout_seconds: float | None = (
      300.0  # Auto-stop after duration, None = no limit
  )

  # If True, keep robot in current mode during prepare and only switch to
  # TEACH/TELEOP when start() is called. If False (default), switch mode
  # immediately in prepare() so user can move robot before recording starts.
  hold_until_start: bool = False


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

  # The visual reference is an AprilTag fiducial marker
  APRILTAG = enum.auto()

  # no visual reference — joint data only
  NONE = enum.auto()


@dataclasses.dataclass
class VisualPoseEntry:
  """A visual pose stored in the library for visual servoing.

  Attributes:
    name: Unique identifier for the pose.
    description: Human-readable description.
    reference_type: Type of visual reference (AR marker or object).
    camera_type: Which camera was used to capture the pose.
    rgb_image: RGB reference image as [H, W, 3] uint8 array.
    depth_image: Depth reference image as [H, W, 1] int16 array.
    reference_mask: Mask indicating reference location as [H, W] array.
    apriltag_metadata: Optional metadata for AprilTag-based poses.
  """

  # Name of the visual pose, must be unique.
  name: str

  # Human readable description of pose. Largely optional.
  description: str

  # What kind of visual reference is used for this pose.
  reference_type: VisualReference

  # Which camera was used to capture the visual pose.
  camera_type: CameraType

  # RGB image from the camera taken during pose definition time.
  # Tensor shape is [H W 3] and dtype np.uint8.
  rgb_image: np.ndarray

  # Depth image from the camera taken during pose definition time.
  # Tensor shape is [H W 1] and dtype np.int16.
  depth_image: np.ndarray

  # Mask for the above RGB and Depth images, defining where the visual reference
  # is the image. Tensor shape is [H W]
  reference_mask: np.ndarray

  # Optional AprilTag metadata for APRILTAG reference types.
  apriltag_metadata: "AprilTagPoseMetadata | None" = None


@dataclasses.dataclass
class ListVisualPosesResponse:
  """Response containing all visual poses in the library.

  Attributes:
    poses: List of visual pose entries.
  """

  poses: list[VisualPoseEntry]


@dataclasses.dataclass
class AddVisualPoseQuery:
  """Query to add a visual pose to the library.

  Attributes:
    pose: The visual pose entry to add.
    allow_overwrite: If True, overwrite existing pose with same name.
  """

  pose: VisualPoseEntry

  # Whether to allow overwriting an existing trajectory with the same name.
  allow_overwrite: bool = False


@dataclasses.dataclass
class AddVisualPoseQueryResponse:
  """Response after attempting to add a visual pose.

  Attributes:
    success: True if the pose was added.
  """

  success: bool


@dataclasses.dataclass
class DeleteVisualPoseQuery:
  """Query to delete a visual pose from the library.

  Attributes:
    pose_name: Name of the pose to delete.
  """

  pose_name: str


@dataclasses.dataclass
class DeleteVisualPoseQueryResponse:
  """Response after attempting to delete a visual pose.

  Attributes:
    success: True if the pose was deleted.
  """

  success: bool


@dataclasses.dataclass
class LoadVisualPoseQuery:
  """Query to load a visual pose from the library.

  Attributes:
    pose_name: Name of the pose to load.
  """

  pose_name: str


@dataclasses.dataclass
class LoadVisualPoseQueryResponse:
  """Response containing the loaded visual pose.

  Attributes:
    pose: The visual pose entry, or None if not found.
  """

  pose: VisualPoseEntry | None


@dataclasses.dataclass
class VisualReferenceSegmentationQuery:
  """Query to segment a visual reference from a single frame.

  Attributes:
    frame: RGB image as [H, W, 3] uint8 array.
    positive_points: Points on the reference as [N, 2] int32 (Y, X).
    negative_points: Points not on the reference as [N, 2] int32 (Y, X).
  """

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
  """Response containing the segmentation mask for the visual reference.

  Attributes:
    segmentation_mask: Binary mask as [H, W] bool array.
  """

  # [H W] shaped tensor of segmentation masks. Dtype is np.bool_.
  segmentation_mask: np.ndarray


################################
# Visual Recording queries     #
################################


@dataclasses.dataclass
class PrepareVisualRecordingQuery:
  """Query to prepare for visual trajectory recording."""

  trajectory_source: TrajectorySource = TrajectorySource.ROBOT
  timeout_seconds: float | None = (
      300.0  # Auto-stop after duration, None = no limit
  )

  # If True, keep robot in current mode during prepare and only switch to
  # TEACH/TELEOP when start() is called. If False (default), switch mode
  # immediately in prepare() so user can move robot before recording starts.
  hold_until_start: bool = False


@dataclasses.dataclass
class PrepareVisualRecordingResponse:
  """Response after preparing for visual recording."""

  error: str | None = None


@dataclasses.dataclass
class StartVisualRecordingResponse:
  """Response after starting visual recording."""

  error: str | None = None


@dataclasses.dataclass
class StopVisualRecordingResponse:
  """Response after stopping visual recording."""

  error: str | None = None
  frame_count: int = 0
  period_seconds: float = 0.0
  joint_positions: np.ndarray | None = None  # [T, 7]


@dataclasses.dataclass
class VisualRecordingStateResponse:
  """Response containing the current visual recording state."""

  is_recording: bool
  sample_count: int
  elapsed_seconds: float
  timed_out: bool
  trajectory_source: TrajectorySource | None = None
  timeout_seconds: float | None = None


@dataclasses.dataclass
class GetVisualRecordingFrameQuery:
  """Query to fetch a single recorded frame."""

  frame_index: int = 0


@dataclasses.dataclass
class GetVisualRecordingFrameResponse:
  """Response containing a single recorded frame."""

  rgb: np.ndarray | None = None  # [H, W, 3]
  depth: np.ndarray | None = None  # [H, W, 1]


@dataclasses.dataclass
class GetVisualRecordingFrameThumbnailsResponse:
  """Response containing thumbnails of all recorded frames."""

  thumbnails: list[np.ndarray]  # List of [H, W, 3] uint8 arrays (small JPEGs)


# Subsample every Nth recorded frame for display and processing.
# At 10Hz recording, step=2 gives ~5Hz effective (0.2s between frames).
DEFAULT_ANNOTATION_SUBSAMPLE = 2


@dataclasses.dataclass
class SegmentVisualRecordingQuery:
  """Query to segment an object in the recorded visual frames using SAM2.

  Unlike ObjectSegmentationQuery, frames are NOT included — the server
  reads them directly from the in-memory recording buffer.

  Attributes:
    positive_points: Points on the object as list of [N, 3] int32 (T, Y, X).
    negative_points: Points not on the object as list of [N, 3] int32 (T, Y, X).
    subsample: Keep every Nth frame for segmentation. 1 = all frames.
  """

  positive_points: list[np.ndarray]
  negative_points: list[np.ndarray]
  subsample: int = DEFAULT_ANNOTATION_SUBSAMPLE
  start_frame: int | None = None
  end_frame: int | None = None


@dataclasses.dataclass
class SegmentVisualRecordingResponse:
  """Response containing segmentation masks for the visual recording."""

  segmentation_mask: np.ndarray  # [T, H, W] bool


@dataclasses.dataclass
class GenerateAprilTagMasksQuery:
  """Query to detect an AprilTag across all recorded frames and generate masks.

  The server reads frames from the in-memory recording buffer, runs AprilTag
  detection on each frame, and generates binary masks from the tag corners.

  Attributes:
    tag_family: The AprilTag family to detect.
    tag_id: The specific tag ID to track.
    tag_size: Physical tag size in meters.
  """

  tag_family: "AprilTagFamily"
  tag_id: int
  tag_size: float
  start_frame: int | None = None
  end_frame: int | None = None


@dataclasses.dataclass
class GenerateAprilTagMasksResponse:
  """Response containing per-frame masks generated from AprilTag corners."""

  segmentation_mask: np.ndarray  # [T, H, W] bool
  apriltag_metadata: "AprilTagPoseMetadata"
  error: str | None = None


@dataclasses.dataclass
class LoadVisualTrajectoryIntoBufferQuery:
  """Query to load a saved trajectory's frames into the recording buffer."""

  name: str


@dataclasses.dataclass
class LoadVisualTrajectoryIntoBufferResponse:
  """Response for loading a trajectory into the recording buffer."""

  success: bool
  reference_masks: np.ndarray | None = None
  reference_type: "VisualReference | None" = None
  num_frames: int = 0


@dataclasses.dataclass
class SaveVisualRecordingQuery:
  """Query to save the current visual recording to the library."""

  name: str
  reference_type: VisualReference
  description: str = ""
  camera_type: CameraType = CameraType.WRIST
  reference_masks: np.ndarray | None = None  # [T, H, W]
  apriltag_metadata: "AprilTagPoseMetadata | None" = None
  allow_overwrite: bool = False
  start_frame: int | None = None
  end_frame: int | None = None


@dataclasses.dataclass
class SaveVisualRecordingResponse:
  """Response after saving a visual recording."""

  error: str | None = None


######################################
# Visual Trajectory Library queries  #
######################################


@dataclasses.dataclass
class VisualTrajectoryLibraryEntry:
  """Entry in the visual trajectory library.

  Combines trajectory joint data with visual frame data for visual-guided
  trajectory execution. All per-frame data is captured at the same sample rate.
  """

  name: str

  description: str

  # Which camera was used to capture frames.
  camera_type: CameraType

  # What kind of visual reference is used for masks.
  reference_type: VisualReference

  # The source of the joint data used to record this trajectory.
  trajectory_source: TrajectorySource

  # The number of seconds the trajectory spans from start to end.
  period_seconds: float

  # RGB frames at each sample. Shape is [N, H, W, 3], dtype uint8.
  rgb_frames: np.ndarray

  # Depth frames at each sample. Shape is [N, H, W, 1], dtype int16.
  depth_frames: np.ndarray

  # Reference masks at each sample. Shape is [N, H, W].
  # All-zero where no mask is annotated.
  reference_masks: np.ndarray

  # Joint absolute positions at each sample. Shape is [N, 7] for joint + gripper.
  joint_positions: np.ndarray

  # Commanded joint absolute positions at each sample. Shape is [N, 7].
  commanded_joint_positions: np.ndarray

  # Joint efforts at each sample. Shape is [N, 7].
  joint_efforts: np.ndarray

  # Wrist cartesian poses at each sample. Shape is [N, 8] for xyz + quaternion + gripper.
  wrist_poses: np.ndarray

  # Optional AprilTag metadata for APRILTAG reference types.
  apriltag_metadata: "AprilTagPoseMetadata | None" = None


@dataclasses.dataclass
class VisualTrajectoryMetadataEntry:
  """Lightweight metadata for a visual trajectory, without heavy arrays."""

  name: str
  description: str
  camera_type: CameraType
  reference_type: VisualReference
  trajectory_source: TrajectorySource
  period_seconds: float
  num_frames: int
  # First RGB frame for preview. Shape [H, W, 3], dtype uint8.
  preview_rgb: np.ndarray
  # First reference mask for preview. Shape [H, W].
  preview_mask: np.ndarray
  apriltag_metadata: "AprilTagPoseMetadata | None" = None


@dataclasses.dataclass
class UpdateVisualTrajectoryMasksQuery:
  """Query to update masks on an existing visual trajectory."""

  name: str
  reference_masks: np.ndarray
  reference_type: VisualReference
  apriltag_metadata: "AprilTagPoseMetadata | None" = None


@dataclasses.dataclass
class UpdateVisualTrajectoryMasksResponse:
  """Response for updating masks."""

  success: bool
  error: str | None = None


@dataclasses.dataclass
class ListVisualTrajectoriesResponse:
  """Response containing all visual trajectories in the library.

  Attributes:
    visual_trajectories: List of visual trajectory metadata entries.
  """

  visual_trajectories: list[VisualTrajectoryMetadataEntry]


@dataclasses.dataclass
class AddVisualTrajectoryQuery:
  """Query to add a visual trajectory to the library.

  Attributes:
    visual_trajectory: The visual trajectory entry to add.
    allow_overwrite: If True, overwrite existing entry with same name.
  """

  visual_trajectory: VisualTrajectoryLibraryEntry
  allow_overwrite: bool = False


@dataclasses.dataclass
class AddVisualTrajectoryQueryResponse:
  """Response after attempting to add a visual trajectory.

  Attributes:
    success: True if the visual trajectory was added.
  """

  success: bool


@dataclasses.dataclass
class DeleteVisualTrajectoryQuery:
  """Query to delete a visual trajectory from the library.

  Attributes:
    visual_trajectory_name: Name of the visual trajectory to delete.
  """

  visual_trajectory_name: str


@dataclasses.dataclass
class DeleteVisualTrajectoryQueryResponse:
  """Response after attempting to delete a visual trajectory.

  Attributes:
    success: True if the visual trajectory was deleted.
  """

  success: bool


@dataclasses.dataclass
class LoadVisualTrajectoryQuery:
  """Query to load a visual trajectory from the library.

  Attributes:
    visual_trajectory_name: Name of the visual trajectory to load.
  """

  visual_trajectory_name: str


@dataclasses.dataclass
class LoadVisualTrajectoryQueryResponse:
  """Response containing the loaded visual trajectory.

  Attributes:
    visual_trajectory: The visual trajectory entry, or None if not found.
  """

  visual_trajectory: VisualTrajectoryLibraryEntry | None


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
  request_data: dict | None = None
  result_data: dict | None = None
  termination_reason: str | None = None
  error_message: str | None = None
  execution_mode_before: str | None = None
  execution_mode_after: str | None = None
  progress_data: dict | None = None


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


@dataclasses.dataclass
class BehaviourReplayNotebookCell:
  """Rendered Python source for replaying one ticket."""

  ticket_id: str
  behaviour_type: str
  supported: bool
  unsupported_reason: str | None = None
  python_source: str | None = None


@dataclasses.dataclass
class ReplayNotebookCellsQuery:
  """Query notebook replay cells for specific tickets."""

  ticket_ids: list[str]


@dataclasses.dataclass
class ReplayNotebookCellsResponse:
  """Notebook replay cells for requested tickets."""

  cells: list[BehaviourReplayNotebookCell]
  missing_ticket_ids: list[str] = dataclasses.field(default_factory=list)


# Behaviour initiation queries


@enum.unique
class TrajectoryMotionType(enum.Enum):
  """How to execute a trajectory motion behaviour."""

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
class VisualPoseMotionQuery:
  """Moves to a visual pose from the visual pose library."""

  visual_pose_name: str
  period_seconds: float


@dataclasses.dataclass
class VisualTrajectoryMotionQuery:
  """Executes a visual trajectory from the visual trajectory library."""

  visual_trajectory_name: str
  period_seconds: float | None = None

  # How to execute the visual trajectory. FULL plays the entire trajectory,
  # GO_TO_START moves to the first frame using visual servoing.
  motion_type: TrajectoryMotionType = TrajectoryMotionType.FULL

  # If this is set to True, then the gripper component of the trajectory is
  # ignored and the gripper position does not change through the trajectory.
  static_gripper: bool = False


@dataclasses.dataclass
class OpenGripperQuery:
  """Open the gripper to a target position."""

  target_position: float = 0.1  # default 10cm open


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
class ExecuteLearnedBehaviorQuery:
  """Execute a learned behaviour via local or remote inference.

  Inference mode selection (in order of priority):
  1. If service_address is set, uses remote inference
  2. If model_id is set and prefer_service=True (default), automatically
     checks for running inference service. Uses remote if found, otherwise
     falls back to local inference
  3. If model_id is set and prefer_service=False, forces local inference
  """

  model_id: str = ""
  service_address: str = ""
  prefer_service: bool = True
  timeout_seconds: float | None = None
  obs_history_len: int = 1
  buffer_actions: int = 20
  action_offset: int = 2
  action_key: str = "action"


@dataclasses.dataclass
class PredictProgressQuery:
  """Predict task completion progress from current camera image.

  Uses a progress prediction model to estimate how close the current
  task is to completion. Returns a value in [0, 1].

  Specify exactly one of model_id (for local inference) or service_address
  (for remote inference).
  """

  model_id: str = ""
  service_address: str = ""


@dataclasses.dataclass
class PredictProgressResponse:
  """Response containing predicted progress value.

  If error is set, progress will be None and should not be used.
  Callers must check error before using the progress value.
  """

  progress: float | None
  error: str | None = None


@dataclasses.dataclass
class GoToNeutralPoseQuery:
  """Move the arm to a neutral pose."""

  pass


@dataclasses.dataclass
class AlignLeaderWithFollowerQuery:
  """Align the leader arm with the follower arm position."""

  timeout_seconds: float = 5.0
  threshold: float = 0.1


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


#############################
# Episode Observer queries  #
#############################


@dataclasses.dataclass
class ComponentHealthStatus:
  """Health status for a single hardware component."""

  name: str
  status: str
  last_update_time: float
  message: str = ""


@dataclasses.dataclass
class HardwareHealthResponse:
  """Aggregated hardware health response."""

  is_healthy: bool
  summary: str
  checked_at_sec: float
  components: list[ComponentHealthStatus]


@enum.unique
class CollectDataPhase(enum.Enum):
  """Backend-owned collect-data workflow phase."""

  IDLE = enum.auto()
  PREPARING = enum.auto()
  READY_FOR_START = enum.auto()
  RECORDING = enum.auto()
  RECORDING_PAUSED = enum.auto()
  PENDING_SAVE = enum.auto()
  ERROR = enum.auto()


@dataclasses.dataclass
class CollectDataPrepareQuery:
  """Query to prepare collect-data recording workflow."""

  continuous_teleop: bool | None = None
  start_trajectory: str | None = None
  align_timeout_seconds: float | None = None
  align_threshold: float | None = None
  behaviour_wait_timeout_seconds: float | None = None


@dataclasses.dataclass
class CollectDataPrepareResponse:
  """Response after preparing collect-data workflow."""

  error: str | None = None


@dataclasses.dataclass
class CollectDataStartResponse:
  """Response after starting collect-data recording."""

  error: str | None = None


@dataclasses.dataclass
class CollectDataStopResponse:
  """Response after stopping collect-data recording."""

  error: str | None = None


@dataclasses.dataclass
class CollectDataSaveQuery:
  """Query to save the current collect-data episode."""

  entry_prefix: str


@dataclasses.dataclass
class CollectDataSaveResponse:
  """Response after saving the current collect-data episode."""

  error: str | None = None


@dataclasses.dataclass
class CollectDataDiscardResponse:
  """Response after discarding the current collect-data episode."""

  error: str | None = None


@dataclasses.dataclass
class CollectDataStateResponse:
  """Response containing the current collect-data state."""

  is_available: bool
  phase: CollectDataPhase
  control_message: str
  fps: float | None
  is_recording: bool
  pending_save_decision: bool
  ready_for_start: bool
  task_description: str
  has_error: bool
  is_human: bool = False
  hardware_healthy: bool = True
  hardware_summary: str = ""


#############################
# DAgger types              #
#############################


@enum.unique
class DaggerPhase(enum.Enum):
  """DAgger workflow phase."""

  INACTIVE = enum.auto()
  ALIGNING = enum.auto()
  ALIGNED = enum.auto()
  TELEOP = enum.auto()
  POLICY = enum.auto()
  ERROR = enum.auto()


@dataclasses.dataclass
class DaggerConfigQuery:
  """Configuration for DAgger policy-assist."""

  service_address: str = ""
  timeout_seconds: float | None = None
  obs_history_len: int = 1
  buffer_actions: int = 20
  action_offset: int = 2
  action_key: str = "action"

  termination_service_address: str = ""
  termination_threshold: float = 0.95
  termination_min_frames: int = 2
  termination_poll_interval_seconds: float = 0.1

  align_timeout_seconds: float = 1.0
  align_threshold: float = 0.1
  behaviour_wait_timeout_seconds: float = 30.0


@dataclasses.dataclass
class DaggerStateResponse:
  """Current DAgger workflow state."""

  phase: DaggerPhase
  control_message: str
  has_error: bool
  error_message: str | None
  policy_ticket_id: str | None
  is_human: bool
  intervention_count: int
  last_progress: float | None
  termination_frames_above: int
  active_source: str
  config: DaggerConfigQuery


@dataclasses.dataclass
class DaggerConfigureResponse:
  """Response after applying DAgger configuration."""

  error: str | None = None


@dataclasses.dataclass
class DaggerToggleResponse:
  """Response after toggling DAgger control."""

  error: str | None = None


@dataclasses.dataclass
class DaggerStopResponse:
  """Response after stopping DAgger control."""

  error: str | None = None


@dataclasses.dataclass
class EpisodeObserverStateResponse:
  """Response containing the current episode observer state for UI display."""

  is_available: bool
  is_recording: bool
  control_message: str
  fps: float | None
  pending_save_decision: bool
  task_description: str
  has_error: bool
  is_human: bool = False  # Current is_human state for DAgger tracking
  hardware_healthy: bool = True
  hardware_summary: str = ""


@dataclasses.dataclass
class EpisodeObserverSaveQuery:
  """Query to save the current episode with an optional entry prefix."""

  entry_prefix: str


@dataclasses.dataclass
class SetTaskDescriptionQuery:
  """Query to set the task description for the current episode."""

  description: str


@dataclasses.dataclass
class SetIsHumanQuery:
  """Query to set the is_human flag for subsequent timesteps.

  When is_human=True, the episode observer will inject is_human=True into
  observations for all timesteps recorded until is_human is set to False.
  This is used for DAgger-style data collection where human interventions
  need to be tracked per-timestep.
  """

  is_human: bool


##############################
# AprilTag Detection queries #
##############################


@enum.unique
class AprilTagFamily(enum.Enum):
  """Supported AprilTag families."""

  TAG16H5 = "tag16h5"
  TAG36H11 = "tag36h11"
  TAG36H10 = "tag36h10"
  TAG25H9 = "tag25h9"
  TAGCIRCLE21H7 = "tagCircle21h7"
  TAGCIRCLE49H12 = "tagCircle49h12"
  TAGCUSTOM48H12 = "tagCustom48h12"
  TAGSTANDARD41H12 = "tagStandard41h12"
  TAGSTANDARD52H13 = "tagStandard52h13"


@dataclasses.dataclass
class AprilTagPoseMetadata:
  """Metadata for AprilTag-based visual pose.

  Attributes:
    tag_family: The AprilTag family used.
    tag_id: The ID of the specific tag.
    tag_size: The physical size of the tag in meters.
  """

  tag_family: AprilTagFamily
  tag_id: int
  tag_size: float


@dataclasses.dataclass
class AprilTagPose:
  """6DoF pose of a detected AprilTag in camera frame.

  The pose is in OpenCV camera convention (Z forward, Y down).
  """

  rotation: np.ndarray  # (3, 3) rotation matrix
  translation: np.ndarray  # (3,) translation vector [x, y, z] in meters


@dataclasses.dataclass
class AprilTagDetection:
  """A single AprilTag detection result."""

  id: int
  family: AprilTagFamily
  hamming: int
  margin: float
  center: np.ndarray  # (2,) - [x, y] pixel coordinates
  corners: np.ndarray  # (4, 2) - [lb, rb, rt, lt] corner coordinates
  pose: AprilTagPose | None = None  # 6DoF pose if intrinsics provided


@dataclasses.dataclass
class AprilTagDetectQuery:
  """Query for AprilTag detection from a provided image.

  Attributes:
    image: RGB image as numpy array with shape (H, W, 3), dtype uint8.
    families: Tag families to detect, or None to detect all supported families.
    intrinsics: Camera intrinsic matrix [3, 3] for pose estimation. If None,
      no pose.
    tag_size: Tag size in meters for pose estimation (required with intrinsics).
  """

  image: np.ndarray
  families: list[AprilTagFamily] | None = None
  intrinsics: np.ndarray | None = None
  tag_size: float | None = None


@dataclasses.dataclass
class AprilTagDetectResponse:
  """Result from AprilTag detection.

  Attributes:
    detections: List of detected AprilTags.
    error: Error message if detection failed, None on success.
  """

  detections: list[AprilTagDetection]
  error: str | None = None


@dataclasses.dataclass
class AprilTagServiceInfoResponse:
  """Information about the AprilTag detection service.

  Attributes:
    available: Whether the service is available.
    model_type: Type of the detection model.
    model_description: Description of the detection model.
    error: Error message if service unavailable.
  """

  available: bool
  model_type: str | None = None
  model_description: str | None = None
  error: str | None = None


####################
# Training queries #
####################


@dataclasses.dataclass
class StartSkillTrainingQuery:
  """Start skill model training.

  Uses entry_filters to automatically build/cache a dataset from the data
  warehouse using the SDK default configuration.

  Attributes:
    model_name: Name for the exported model in the model warehouse.
    training_steps: Total number of training steps to run.
    entry_filters: List of glob patterns for selecting entries from data
      warehouse (e.g., ["pick_up_can*", "open_door*"]). Multiple patterns
      are combined.
    force_rebuild: If True, force rebuild dataset even if cached version exists.
    batch_size: Batch size for training.
    prediction_horizon: Number of future steps to predict.
    enable_advantage_weighting: Enable CRR-style advantage weighting.
    value_function_model_id: Model warehouse ID for the frozen V(s).
    crr_type: CRR weighting mode: "hard", "soft", or "soft_cutoff".
    crr_negative_weight: Weight for negative advantage (soft_cutoff mode).
    advantage_gae_lambda: GAE lambda (0=1-step, 1=H-step, 0.95=blended).
    advantage_cutoff: Advantage threshold for sample acceptance.
  """

  model_name: str
  training_steps: int
  # List of glob patterns for selecting entries from data warehouse
  entry_filters: list[str] = dataclasses.field(default_factory=list)
  model_save_dir: str = ""
  # If True, force rebuild dataset even if cached version exists.
  force_rebuild: bool = False
  # Training configuration
  batch_size: int = 64
  prediction_horizon: int = 32
  use_joint_torques: bool = False  # Include piper_joint_torques in proprio
  # Checkpoint configuration
  checkpoint_interval_steps: int = 1000  # Save checkpoint every N steps
  max_checkpoints_to_keep: int = 10  # Keep 10 most recent checkpoints
  # CRR advantage weighting
  enable_advantage_weighting: bool = False
  value_function_model_id: str = ""
  crr_type: str = "hard"  # "hard", "soft", or "soft_cutoff"
  crr_negative_weight: float = 0.1
  advantage_gae_lambda: float = 0.95
  advantage_cutoff: float = -0.02


@dataclasses.dataclass
class StartSkillTrainingResponse:
  """Response when skill training is started.

  Use get_training_status() to monitor progress and get phase details.
  """

  error: str | None = None


@dataclasses.dataclass
class TrainingStatusResponse:
  """Response containing training status information."""

  is_finished: bool
  steps_completed: int
  max_steps: int
  loss: float
  fps: float  # Steps per second
  seconds_per_step: float
  metrics: dict[str, float] | None = None  # Additional metrics from training
  # Phase: "idle", "preparing_dataset", "training", "finished", "failed"
  phase: str = "idle"
  export_entries_processed: int = 0  # Number of entries exported so far
  export_entries_total: int = 0  # Total entries to export
  # Training configuration - populated when training starts, persists after
  # training finishes, cleared on hard reset. None if trainer never used or
  # after hard reset.
  model_name: str | None = None
  entry_filters: list[str] | None = None
  batch_size: int | None = None
  prediction_horizon: int | None = None


@dataclasses.dataclass
class CancelTrainingQuery:
  """Query to cancel training."""

  pass


@dataclasses.dataclass
class CancelTrainingResponse:
  """Result of a cancel training request."""

  success: bool
  error: str | None = None


@dataclasses.dataclass
class ResetTrainerResponse:
  """Result of a reset trainer request."""

  success: bool
  error: str | None = None


@dataclasses.dataclass
class StartExportQuery:
  """Query to start async model export.

  Attributes:
    checkpoint_step: Export model from this checkpoint step. If None, uses the
      latest checkpoint. Must match an existing checkpoint step.
  """

  checkpoint_step: int | None = None  # None = latest checkpoint
  model_name: str | None = None  # None = the most recent/current model trained.
  # List of glob patterns for selecting entries from data warehouse. This is
  # required if you specify model_name.
  entry_filters: list[str] = dataclasses.field(default_factory=list)
  # The following parameters should be specified if the model was trained with
  # a custom setting that differs from the defaults in StartSkillTrainingQuery,
  # to ensure the model is exported correctly.
  # If None, defaults to the same values as used in StartSkillTrainingQuery.
  model_save_dir: str | None = None
  prediction_horizon: int | None = None
  use_joint_torques: bool | None = None


@dataclasses.dataclass
class StartExportResponse:
  """Response when export is started."""

  error: str | None = None
  # Available checkpoints if the requested step was not found
  available_checkpoints: list[int] | None = None


@dataclasses.dataclass
class ExportStatusResponse:
  """Response containing export status."""

  is_exporting: bool
  is_finished: bool
  error: str | None = None
  model_id: str | None = None  # Set when export completes successfully
  checkpoint_step: int | None = None  # The checkpoint being exported


@dataclasses.dataclass
class ListCheckpointsResponse:
  """Response containing available checkpoint steps."""

  checkpoint_steps: list[int]  # Available checkpoint steps, sorted ascending
  error: str | None = None


@dataclasses.dataclass
class ListEntryFiltersQuery:
  """Query to list entry filter IDs from the data warehouse."""

  search: str = ""


@dataclasses.dataclass
class ListEntryFiltersResponse:
  """Response containing available entry filter IDs."""

  success: bool
  filters: list[str] = dataclasses.field(default_factory=list)
  error: str | None = None


#####################################
# Progress prediction training APIs #
#####################################


@dataclasses.dataclass
class StartProgressTrainingQuery:
  """Start progress prediction model training.

  Attributes:
    model_name: Name for the exported model in the model warehouse.
    entry_filters: Glob patterns for full episode entries from data warehouse
      (e.g., ["pick_up_can*", "place_object*"]). Processes entire episodes.
    human_entry_filters: Glob patterns for human demonstration entries
      (e.g., ["dagger_*"]). Extracts only the human segments from episodes.
    training_steps: Total number of training steps to run.
    force_rebuild: If True, rebuild the dataset even if a fresh cache exists.
    batch_size: Training batch size.
    task_type: "classification" for binary done/not-done, "regression" for
      continuous 0-1 progress.
    cameras: Camera names to use (e.g., ["wrist_camera"]).
    resume_from: Checkpoint ID to resume from (e.g., "progress_model/20260202-150000").
      If None, starts fresh training with a new checkpoint directory.
    checkpoint_interval_steps: Save checkpoint every N steps. Default 1000.
    max_checkpoints_to_keep: Max checkpoints to keep. Default 10.
  """

  model_name: str
  training_steps: int
  entry_filters: list[str] | None = None
  human_entry_filters: list[str] | None = None
  force_rebuild: bool = False
  batch_size: int = 32
  task_type: str = "classification"  # "classification" or "regression"
  cameras: list[str] | None = None
  resume_from: str | None = None
  checkpoint_interval_steps: int = 1000  # Save checkpoint every N steps
  max_checkpoints_to_keep: int = 10  # Keep 10 most recent checkpoints


@dataclasses.dataclass
class StartProgressTrainingResponse:
  """Response when progress prediction training is started.

  Attributes:
    error: Error message if training could not be started, None on success.
    dataset_was_rebuilt: True if the dataset was built/rebuilt for this request.
    dataset_is_stale: True if using stale cached data.
    cached_entry_count: Number of entries in the cached dataset.
    current_entry_count: Current number of matching entries in data warehouse.
  """

  error: str | None = None
  dataset_was_rebuilt: bool = False
  dataset_is_stale: bool = False
  cached_entry_count: int | None = None
  current_entry_count: int | None = None


@dataclasses.dataclass
class ProgressTrainingStatusResponse:
  """Response containing progress prediction training status."""

  is_finished: bool
  phase: str  # idle, preparing_dataset, training, finished, failed
  steps_completed: int
  max_steps: int
  loss: float
  fps: float  # Steps per second
  seconds_per_step: float
  accuracy: float | None = None  # Classification accuracy (if applicable)
  f1: float | None = None  # F1 score (if applicable)
  val_loss: float | None = None  # Validation loss
  val_accuracy: float | None = None  # Validation accuracy
  val_f1: float | None = None  # Validation F1 score
  checkpoint_id: str | None = None  # e.g., "progress_model/20260202-150000"
  error: str | None = None  # Error message if phase is "failed"
  export_entries_processed: int = 0  # Number of entries exported so far
  export_entries_total: int = 0  # Total entries to export
  # Training configuration - for UI auto-fill on reconnect
  model_name: str | None = None
  entry_filters: list[str] | None = None
  batch_size: int | None = None
  task_type: str | None = None


@dataclasses.dataclass
class CancelProgressTrainingQuery:
  """Query to cancel progress prediction training."""

  pass


@dataclasses.dataclass
class CancelProgressTrainingResponse:
  """Result of cancelling progress prediction training."""

  success: bool
  error: str | None = None


#########################
# Model Service queries #
#########################


@dataclasses.dataclass
class StartModelServiceQuery:
  """Query to start an inference service for a model.

  Attributes:
    model_id: The model warehouse model ID to serve.
    port: Optional port to use. If None, a port is auto-assigned.
  """

  model_id: str
  port: int | None = None


@dataclasses.dataclass
class StartModelServiceResponse:
  """Response from starting an inference service.

  Attributes:
    address: The service address (e.g., "tcp://localhost:4601").
  """

  address: str


@dataclasses.dataclass
class StopModelServiceQuery:
  """Query to stop an inference service.

  Attributes:
    model_id: The model ID of the service to stop.
  """

  model_id: str


@dataclasses.dataclass
class ModelServiceInfo:
  """Info about a running inference service.

  Attributes:
    model_id: The model warehouse model ID being served.
    address: The service address.
    healthy: Whether the service is responding to health checks.
  """

  model_id: str
  address: str
  healthy: bool


@dataclasses.dataclass
class ListModelServicesResponse:
  """Response listing all running inference services.

  Attributes:
    services: List of running service info.
  """

  services: list[ModelServiceInfo]


@dataclasses.dataclass
class WaitModelServicesQuery:
  """Query to wait for model services to become ready."""

  model_ids: list[str] | None = None  # None = all services
  timeout: float = 120.0
  poll_interval: float = 1.0


@dataclasses.dataclass
class WaitModelServicesResponse:
  """Response from waiting for model services."""

  success: bool  # True if all became ready, False on timeout
  ready_models: list[str]  # Models that became ready
  pending_models: list[str]  # Models still not ready (if timeout)
