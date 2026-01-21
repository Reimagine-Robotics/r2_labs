"""High-level clients for robot control and behaviour execution."""

import dataclasses
import pickle
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from r2_labs.rpc import client
from r2_labs.sdk import futures as sdk_futures
from r2_labs.sdk import rpc_api


def _rpc_call(
    rpc_client: client.BaseClient,
    fn_name: str,
    data: Any | None = None,
    timeout: int | None = None,
) -> Any:
  """Make an RPC call with automatic serialization.

  Args:
    rpc_client: The RPC client to use.
    fn_name: Name of the remote function.
    data: Data to serialize and send, or None for no-arg calls.
    timeout: Optional timeout override in milliseconds.

  Returns:
    The deserialized response from the server.
  """
  if data is None:
    serialized_result = rpc_client(fn_name=fn_name, timeout=timeout)
  else:
    serialized_result = rpc_client(
        fn_name=fn_name, data=pickle.dumps(data), timeout=timeout
    )

  assert isinstance(serialized_result, bytes)
  return pickle.loads(serialized_result)


def _with_buffer(timeout_seconds: float) -> float:
  """Add a small buffer to absorb RPC/polling latency."""
  return timeout_seconds + 1.0


@dataclasses.dataclass(frozen=True)
class ObjectAnnotationPoint:
  """Point annotation for object segmentation over a frame sequence.

  Attributes:
    x: Column index of the annotated pixel.
    y: Row index of the annotated pixel.
    frame_index: Frame index in the input sequence.
    label: 1 for positive points, 0 for negative points.
  """

  x: int
  y: int
  frame_index: int
  label: int


@dataclasses.dataclass(frozen=True)
class AprilTagCameraDetection:
  """AprilTag detection results alongside the captured camera data.

  Attributes:
    camera_data: Raw camera data used for detection.
    detections: AprilTag detection response for the captured frame.
  """

  camera_data: rpc_api.CameraQueryResponse
  detections: rpc_api.AprilTagDetectResponse


class ExecModeClient:
  """Client for managing robot execution mode (STOP, READY, TEACH, TELEOP)."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def get_execution_mode(self) -> rpc_api.ExecutionModeQueryResponse:
    """Get the current execution mode."""
    result = _rpc_call(
        self._rpc_client, "exec_mode", rpc_api.ExecutionModeQuery(new_mode=None)
    )
    assert isinstance(result, rpc_api.ExecutionModeQueryResponse)
    return result

  def set_execution_mode(
      self, new_mode: rpc_api.ExecutionMode
  ) -> rpc_api.ExecutionModeQueryResponse:
    """Set the execution mode.

    Args:
      new_mode: Target execution mode.

    Returns:
      Response containing the new current mode.
    """
    query = rpc_api.ExecutionModeQuery(new_mode=new_mode)
    result = _rpc_call(self._rpc_client, "exec_mode", query)
    assert isinstance(result, rpc_api.ExecutionModeQueryResponse)
    return result


class RawRobotClient:
  """Client for accessing raw robot sensor data (cameras, proprioception)."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def get_camera_data(
      self,
      camera: rpc_api.CameraType,
  ) -> rpc_api.CameraQueryResponse:
    """Get RGB and depth data from a camera.

    Args:
      camera: Camera to read from.

    Returns:
      Response containing RGB, depth, and intrinsics data.
    """
    query = rpc_api.CameraQuery(camera=camera)
    result = _rpc_call(self._rpc_client, "raw_robot.get_camera_data", query)
    assert isinstance(result, rpc_api.CameraQueryResponse)
    return result

  def get_proprio_data(self) -> rpc_api.ArmStateQueryResponse:
    """Get proprioceptive data (joint positions, velocities, efforts)."""
    result = _rpc_call(self._rpc_client, "raw_robot.get_proprio_data")
    assert isinstance(result, rpc_api.ArmStateQueryResponse)
    return result

  def get_cuff_buttons(self) -> rpc_api.CuffBottonsQueryResponse:
    """Get the current state of the arm cuff buttons."""
    result = _rpc_call(self._rpc_client, "raw_robot.get_cuff_buttons")
    assert isinstance(result, rpc_api.CuffBottonsQueryResponse)
    return result


class QueryClient:
  """Client for synchronous query operations (e.g., object visibility)."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def can_see_object(
      self,
      object_names: Sequence[str],
      timeout_seconds: float = 30.0,
  ) -> rpc_api.CanSeeObjectResponse:
    """Check if any of the specified objects are visible.

    Args:
      object_names: Names of objects to look for.
      timeout_seconds: Maximum time to wait for detection.

    Returns:
      Response indicating visibility and detected object position.
    """
    query = rpc_api.CanSeeObjectQuery(
        object_names=list(object_names),
        timeout_seconds=timeout_seconds,
    )
    timeout = int(_with_buffer(query.timeout_seconds) * 1000)
    result = _rpc_call(
        self._rpc_client,
        "query.can_see_object",
        query,
        timeout=timeout,
    )
    assert isinstance(result, rpc_api.CanSeeObjectResponse)
    return result


class RecordingClient:
  """Client for trajectory recording operations.

  Usage:
    # 1. Prepare for recording (sets trajectory type and execution mode)
    robot.recording.prepare()

    # 2. Start recording (or press cuff button D)
    robot.recording.start()

    # 3. Stop recording and get trajectory (or press cuff button D, or wait
    #    for timeout). This works regardless of how recording was stopped.
    response = robot.recording.stop()
    trajectory = response.trajectory

    # 4. Optionally save the trajectory to the library
    robot.trajectory_library.add_entry(trajectory=trajectory)
  """

  def __init__(self, rpc_client: client.BaseClient) -> None:
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def prepare(
      self,
      trajectory_type: rpc_api.TrajectoryType = (
          rpc_api.TrajectoryType.JOINT_ABSOLUTE
      ),
      trajectory_source: rpc_api.TrajectorySource = (
          rpc_api.TrajectorySource.ROBOT
      ),
      timeout_seconds: float | None = 300.0,
      hold_until_start: bool = False,
  ) -> rpc_api.PrepareRecordingResponse:
    """Prepare for recording with specified trajectory type and execution mode.

    This clears any previously recorded trajectory. The robot will be switched
    to the specified execution mode (TEACH or TELEOP) unless hold_until_start
    is True, in which case the mode change is deferred until start() is called.

    Args:
      trajectory_type: Trajectory type to record.
      trajectory_source: Source of the trajectory data.
      timeout_seconds: Auto-stop after duration, or None to disable.
      hold_until_start: If True, keep robot in current mode during prepare and
        only switch to TEACH/TELEOP when start() is called. If False (default),
        switch mode immediately so user can move robot before recording starts.

    Returns:
      Response with error field set if preparation failed.
    """
    query = rpc_api.PrepareRecordingQuery(
        trajectory_type=trajectory_type,
        trajectory_source=trajectory_source,
        timeout_seconds=timeout_seconds,
        hold_until_start=hold_until_start,
    )
    result = _rpc_call(self._rpc_client, "recording.prepare", query)
    assert isinstance(result, rpc_api.PrepareRecordingResponse)
    return result

  def start(self) -> rpc_api.StartRecordingResponse:
    """Start recording samples.

    Must call prepare() first. Recording can also be started by pressing
    cuff button D on the robot.

    Returns:
      Response with error field set if start failed.
    """
    result = _rpc_call(self._rpc_client, "recording.start")
    assert isinstance(result, rpc_api.StartRecordingResponse)
    return result

  def stop(self) -> rpc_api.StopRecordingResponse:
    """Stop recording and return the recorded trajectory.

    This method is idempotent: if recording was already stopped (e.g., by cuff
    button or timeout), it still returns the trajectory. The trajectory remains
    available until the next prepare() call.

    Returns:
      Response containing the recorded trajectory, or error field if failed.
    """
    result = _rpc_call(self._rpc_client, "recording.stop")
    assert isinstance(result, rpc_api.StopRecordingResponse)
    return result

  def get_state(self) -> rpc_api.RecordingStateResponse:
    """Get the current recording state.

    Returns:
      Current state including is_recording, sample_count, elapsed time, etc.
    """
    result = _rpc_call(self._rpc_client, "recording.get_state")
    assert isinstance(result, rpc_api.RecordingStateResponse)
    return result


class EpisodeObserverClient:
  """Client for episode recording observer control (data gathering UI)."""

  def __init__(self, rpc_client: client.BaseClient) -> None:
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def start(self) -> None:
    """Start episode recording."""
    _rpc_call(self._rpc_client, "episode_observer.start")

  def stop(self) -> None:
    """Stop episode recording."""
    _rpc_call(self._rpc_client, "episode_observer.stop")

  def save(self, entry_prefix: str | None = None) -> None:
    """Save the current episode.

    Args:
      entry_prefix: Optional entry prefix for saving the episode.
    """
    query = rpc_api.EpisodeObserverSaveQuery(entry_prefix=entry_prefix)
    _rpc_call(self._rpc_client, "episode_observer.save", query)

  def discard(self) -> None:
    """Discard the current episode."""
    _rpc_call(self._rpc_client, "episode_observer.discard")

  def get_state(self) -> rpc_api.EpisodeObserverStateResponse:
    """Get the current episode observer state."""
    result = _rpc_call(self._rpc_client, "episode_observer.get_state")
    assert isinstance(result, rpc_api.EpisodeObserverStateResponse)
    return result

  def set_task_description(self, description: str) -> None:
    """Set the task description for the current episode.

    Args:
      description: Task description text.
    """
    query = rpc_api.SetTaskDescriptionQuery(description=description)
    _rpc_call(self._rpc_client, "episode_observer.set_task_description", query)


class ObjectLibraryClient:
  """Client for managing the object library used for detection and grasping."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def list_entries(self) -> rpc_api.ListObjectsResponse:
    """List all objects in the library."""
    result = _rpc_call(self._rpc_client, "object_library.list_entries")
    assert isinstance(result, rpc_api.ListObjectsResponse)
    return result

  def delete_entry(self, object_name: str) -> rpc_api.DeleteObjectQueryResponse:
    """Delete an object from the library.

    Args:
      object_name: Name of the object to delete.
    """
    entry = rpc_api.DeleteObjectQuery(object_name=object_name)
    result = _rpc_call(self._rpc_client, "object_library.delete_entry", entry)
    assert isinstance(result, rpc_api.DeleteObjectQueryResponse)
    return result

  def segment_object(
      self,
      frames: np.ndarray,
      positive_points: Sequence[np.ndarray] | None = None,
      negative_points: Sequence[np.ndarray] | None = None,
      timeout: int | None = None,
  ) -> rpc_api.ObjectSegmentationQueryResponse:
    """Segment an object from video frames using point prompts.

    Args:
      frames: RGB video frames as [T, H, W, 3] uint8 array.
      positive_points: List of [N, 3] points (T, Y, X) on the object.
      negative_points: List of [N, 3] points (T, Y, X) off the object.
      timeout: RPC timeout in milliseconds, or None for default.
    """
    query = rpc_api.ObjectSegmentationQuery(
        frames=frames,
        positive_points=list(positive_points or []),
        negative_points=list(negative_points or []),
    )
    result = _rpc_call(
        self._rpc_client, "object_library.segment_object", query, timeout
    )
    assert isinstance(result, rpc_api.ObjectSegmentationQueryResponse)
    return result

  def segment_object_from_annotations(
      self,
      frames: np.ndarray,
      annotations: Sequence[ObjectAnnotationPoint],
      timeout: int | None = None,
  ) -> rpc_api.ObjectSegmentationQueryResponse:
    """Segment an object using a flat list of point annotations.

    This helper mirrors the HRI REST API payload by converting point
    annotations into the RPC query format.

    Args:
      frames: RGB video frames as [T, H, W, 3] uint8 array.
      annotations: Sequence of point annotations with frame indices and labels.
      timeout: RPC timeout in milliseconds, or None for default.

    Returns:
      Response containing segmentation masks for the queried object.
    """
    positive_points: list[np.ndarray] = []
    negative_points: list[np.ndarray] = []
    for annotation in annotations:
      if annotation.label not in (0, 1):
        raise ValueError(
            "annotation label must be 0 (negative) or 1 (positive)"
        )
      point = np.array(
          [[annotation.frame_index, annotation.y, annotation.x]],
          dtype=np.int32,
      )
      if annotation.label == 1:
        positive_points.append(point)
      else:
        negative_points.append(point)

    return self.segment_object(
        frames=frames,
        positive_points=positive_points,
        negative_points=negative_points,
        timeout=timeout,
    )

  def add_object_views(
      self,
      object_name: str,
      frames: np.ndarray,
      segmentation_mask: np.ndarray,
      object_description: str = "",
      timeout: int | None = None,
  ) -> rpc_api.AddObjectViewsQueryResponse:
    """Add views of an object to the library for recognition training.

    Args:
      object_name: Name of the object (creates new or updates existing).
      frames: RGB video frames as [T, H, W, 3] uint8 array.
      segmentation_mask: Object masks as [T, H, W] array.
      object_description: Human-readable description (ignored if empty).
      timeout: RPC timeout in milliseconds, or None for default.
    """
    query = rpc_api.AddObjectViewsQuery(
        object_name=object_name,
        object_description=object_description,
        frames=frames,
        segmentation_mask=segmentation_mask,
    )
    result = _rpc_call(
        self._rpc_client, "object_library.add_object_views", query, timeout
    )
    assert isinstance(result, rpc_api.AddObjectViewsQueryResponse)
    return result

  def get_heatmap(
      self,
      object_name: str,
      timeout: int | None = None,
  ) -> rpc_api.ObjectHeatmapResponse:
    """Get a live detection heatmap for an object.

    Args:
      object_name: Name of the object to visualize.
      timeout: RPC timeout in milliseconds, or None for default.
    """
    query = rpc_api.ObjectHeatmapQuery(object_name=object_name)
    result = _rpc_call(
        self._rpc_client, "object_library.get_heatmap", query, timeout
    )
    assert isinstance(result, rpc_api.ObjectHeatmapResponse)
    return result


class TrajectoryLibraryClient:
  """Client for managing stored trajectories for replay."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def list_entries(self) -> rpc_api.ListTrajectoriesResponse:
    """List all trajectories in the library."""
    result = _rpc_call(self._rpc_client, "trajectory_library.list_entries")
    assert isinstance(result, rpc_api.ListTrajectoriesResponse)
    return result

  def add_entry(
      self,
      trajectory: rpc_api.TrajectoryLibraryEntry,
      allow_overwrite: bool = False,
  ) -> rpc_api.AddTrajectoryQueryResponse:
    """Add a trajectory to the library.

    Args:
      trajectory: Trajectory entry to add.
      allow_overwrite: Whether to overwrite existing trajectory with same name.
    """
    entry = rpc_api.AddTrajectoryQuery(
        trajectory=trajectory,
        allow_overwrite=allow_overwrite,
    )
    result = _rpc_call(self._rpc_client, "trajectory_library.add_entry", entry)
    assert isinstance(result, rpc_api.AddTrajectoryQueryResponse)
    return result

  def delete_entry(
      self, trajectory_name: str
  ) -> rpc_api.DeleteTrajectoryQueryResponse:
    """Delete a trajectory from the library.

    Args:
      trajectory_name: Name of the trajectory to delete.
    """
    entry = rpc_api.DeleteTrajectoryQuery(trajectory_name=trajectory_name)
    result = _rpc_call(
        self._rpc_client, "trajectory_library.delete_entry", entry
    )
    assert isinstance(result, rpc_api.DeleteTrajectoryQueryResponse)
    return result

  def load_entry(
      self, trajectory_name: str
  ) -> rpc_api.LoadTrajectoryQueryResponse:
    """Load a trajectory from the library by name.

    Args:
      trajectory_name: Name of the trajectory to load.
    """
    query = rpc_api.LoadTrajectoryQuery(trajectory_name=trajectory_name)
    result = _rpc_call(self._rpc_client, "trajectory_library.load_entry", query)
    assert isinstance(result, rpc_api.LoadTrajectoryQueryResponse)
    return result


class VisualPoseLibraryClient:
  """Client for managing visual poses used for visual servoing."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def list_entries(self) -> rpc_api.ListVisualPosesResponse:
    """List all visual poses in the library."""
    result = _rpc_call(self._rpc_client, "visual_pose_library.list_entries")
    assert isinstance(result, rpc_api.ListVisualPosesResponse)
    return result

  def add_entry(
      self,
      pose: rpc_api.VisualPoseEntry,
      allow_overwrite: bool = False,
  ) -> rpc_api.AddVisualPoseQueryResponse:
    """Add a visual pose to the library.

    Args:
      pose: Visual pose entry to add.
      allow_overwrite: Whether to overwrite existing pose with same name.
    """
    entry = rpc_api.AddVisualPoseQuery(
        pose=pose,
        allow_overwrite=allow_overwrite,
    )
    result = _rpc_call(self._rpc_client, "visual_pose_library.add_entry", entry)
    assert isinstance(result, rpc_api.AddVisualPoseQueryResponse)
    return result

  def delete_entry(
      self, pose_name: str
  ) -> rpc_api.DeleteVisualPoseQueryResponse:
    """Delete a visual pose from the library.

    Args:
      pose_name: Name of the pose to delete.
    """
    entry = rpc_api.DeleteVisualPoseQuery(pose_name=pose_name)
    result = _rpc_call(
        self._rpc_client, "visual_pose_library.delete_entry", entry
    )
    assert isinstance(result, rpc_api.DeleteVisualPoseQueryResponse)
    return result

  def load_entry(self, pose_name: str) -> rpc_api.LoadVisualPoseQueryResponse:
    """Load a visual pose from the library by name.

    Args:
      pose_name: Name of the pose to load.
    """
    query = rpc_api.LoadVisualPoseQuery(pose_name=pose_name)
    result = _rpc_call(
        self._rpc_client, "visual_pose_library.load_entry", query
    )
    assert isinstance(result, rpc_api.LoadVisualPoseQueryResponse)
    return result

  def segment_reference(
      self,
      frame: np.ndarray,
      positive_points: np.ndarray | None = None,
      negative_points: np.ndarray | None = None,
      timeout: int | None = None,
  ) -> rpc_api.VisualReferenceSegmentationQueryResponse:
    """Segment a visual reference from a single frame using point prompts.

    Args:
      frame: RGB image as [H, W, 3] uint8 array.
      positive_points: Points on the reference as [N, 2] int32 (Y, X).
      negative_points: Points not on the reference as [N, 2] int32 (Y, X).
      timeout: RPC timeout in milliseconds, or None for default.
    """
    query = rpc_api.VisualReferenceSegmentationQuery(
        frame=frame,
        positive_points=(
            positive_points
            if positive_points is not None
            else np.zeros((0, 2), dtype=np.int32)
        ),
        negative_points=(
            negative_points
            if negative_points is not None
            else np.zeros((0, 2), dtype=np.int32)
        ),
    )
    result = _rpc_call(
        self._rpc_client,
        "visual_pose_library.segment_reference",
        query,
        timeout,
    )
    assert isinstance(result, rpc_api.VisualReferenceSegmentationQueryResponse)
    return result


class AprilTagClient:
  """Client for AprilTag detection operations.

  Usage:
    # Get camera image first
    camera_data = robot.raw_robot.get_camera_data(rpc_api.CameraType.WRIST)

    # Detect AprilTags with pose estimation (5cm tags)
    response = robot.apriltag.detect(
        image=camera_data.rgb,
        families=[rpc_api.AprilTagFamily.TAG36H11],
        intrinsics=camera_data.intrinsics,
        tag_size=0.05,
    )
    for detection in response.detections:
        print(f"Tag {detection.id}: center={detection.center}")
        if detection.pose:
            print(f"  Translation: {detection.pose.translation}")
  """

  def __init__(self, rpc_client: client.BaseClient) -> None:
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def detect(
      self,
      image: np.ndarray,
      families: Sequence[rpc_api.AprilTagFamily] | None = None,
      intrinsics: np.ndarray | None = None,
      tag_size: float | None = None,
      timeout: int | None = None,
  ) -> rpc_api.AprilTagDetectResponse:
    """Detect AprilTags in a provided image.

    Args:
      image: RGB image as [H, W, 3] uint8 array.
      families: Tag families to detect, or None to detect all families.
      intrinsics: Camera intrinsic matrix for pose estimation.
      tag_size: Tag size in meters for pose estimation.
      timeout: Optional RPC timeout in milliseconds.

    Returns:
      Response containing list of detected AprilTags.
    """
    query = rpc_api.AprilTagDetectQuery(
        image=image,
        families=list(families) if families is not None else None,
        intrinsics=intrinsics,
        tag_size=tag_size,
    )
    result = _rpc_call(self._rpc_client, "apriltag.detect", query, timeout)
    assert isinstance(result, rpc_api.AprilTagDetectResponse)
    return result

  def get_service_info(
      self, timeout: int | None = None
  ) -> rpc_api.AprilTagServiceInfoResponse:
    """Get information about the AprilTag detection service.

    Args:
      timeout: Optional RPC timeout in milliseconds.

    Returns:
      Response containing service availability and model info.
    """
    result = _rpc_call(
        self._rpc_client, "apriltag.get_service_info", timeout=timeout
    )
    assert isinstance(result, rpc_api.AprilTagServiceInfoResponse)
    return result


class BehaviourClient:
  """Client for executing robot behaviours asynchronously.

  Behaviours are executed via a ticket system: initiate methods return a
  ticket ID, and the client polls for completion. Convenience methods return
  futures that handle polling automatically.
  """

  def __init__(self, rpc_client_factory: Callable[[], client.BaseClient]):
    """Initialize the client.

    Args:
      rpc_client_factory: Factory to create RPC clients (one per thread).
    """
    self._rpc_client_factory: Callable[[], client.BaseClient] = (
        rpc_client_factory
    )
    self._thread_local_client: threading.local = threading.local()
    self._executor: sdk_futures.SingleThreadExecutor = (
        sdk_futures.SingleThreadExecutor()
    )

  def _get_rpc_client(self) -> client.BaseClient:
    """Get or create a thread-local RPC client."""
    rpc_client = getattr(self._thread_local_client, "rpc_client", None)
    if rpc_client is None:
      rpc_client = self._rpc_client_factory()
      self._thread_local_client.rpc_client = rpc_client
    return rpc_client

  def _submit_behaviour(
      self,
      initiate_fn: Callable[[], rpc_api.BehaviourInitiatedResponse],
      timeout: float | None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
      behaviour_type: str = "behaviour",
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Submit a behaviour for async execution and return a future.

    Args:
      initiate_fn: Function that initiates the behaviour and returns ticket.
      timeout: Maximum time to wait for completion, or None for no limit.
      arm: Which arm(s) this behaviour requires.
      behaviour_type: Name for error messages.

    Returns:
      A future that resolves to the ticket status on completion.
    """
    cancel_event = threading.Event()
    ticket_holder: dict[str, str | None] = {"ticket_id": None}

    def _task() -> rpc_api.TicketStatusResponse:
      try:
        response = initiate_fn()
      except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"{behaviour_type} failed to start") from exc

      if response.error:
        if response.ticket_id:
          ticket_holder["ticket_id"] = response.ticket_id
          try:
            # server may already have a ticket in FAILED with details:
            # surface that
            return self.wait_for_ticket(
                response.ticket_id,
                timeout=timeout,
                cancel_event=cancel_event,
                on_cancel=lambda: self.cancel_behaviour(response.ticket_id)
                and None,
            )
          except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"{behaviour_type} failed to start") from exc
        raise RuntimeError(response.error)

      ticket_holder["ticket_id"] = response.ticket_id
      try:
        return self.wait_for_ticket(
            response.ticket_id,
            timeout=timeout,
            cancel_event=cancel_event,
            on_cancel=lambda: self.cancel_behaviour(response.ticket_id)
            and None,
        )
      except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            f"{behaviour_type} failed while waiting for ticket"
        ) from exc

    def _cancel_callback() -> None:
      cancel_event.set()
      if ticket_holder["ticket_id"] is not None:
        try:
          self.cancel_behaviour(ticket_holder["ticket_id"])
        except Exception:  # pylint: disable=broad-except
          pass

    return self._executor.submit_for_arm(
        arm, _task, cancel_callback=_cancel_callback
    )

  def cancel_behaviour(self, ticket_id: str) -> rpc_api.CancelTicketResponse:
    """Cancel a running behaviour by ticket ID.

    Args:
      ticket_id: Ticket ID to cancel.
    """
    query = rpc_api.CancelTicketQuery(ticket_id=ticket_id)
    result = _rpc_call(self._get_rpc_client(), "behaviour.cancel_ticket", query)
    assert isinstance(result, rpc_api.CancelTicketResponse)
    return result

  def wait_for_ticket(
      self,
      ticket_id: str,
      poll_interval: float = 0.1,
      timeout: float | None = None,
      cancel_event: threading.Event | None = None,
      on_cancel: Callable[[], None] | None = None,
  ) -> rpc_api.TicketStatusResponse:
    """Poll until ticket is COMPLETED or FAILED, or timeout.

    Args:
      ticket_id: The ticket ID to wait for.
      poll_interval: Seconds between status checks.
      timeout: Maximum seconds to wait, or None for no limit.
      cancel_event: Event that triggers cancellation when set.
      on_cancel: Callback invoked when cancel_event is set.

    Raises:
      ValueError: If the ticket is not found.
      TimeoutError: If timeout is reached before completion.
    """
    start = time.time()
    cancel_called = False
    while True:
      if (
          cancel_event is not None
          and cancel_event.is_set()
          and not cancel_called
      ):
        if on_cancel is not None:
          try:
            on_cancel()
          except Exception:  # pylint: disable=broad-except
            pass
        cancel_called = True
      status = self.get_ticket_status(ticket_id)
      if status.not_found:
        raise ValueError(f"ticket not found: {ticket_id}")
      if status.info is not None and status.info.status in (
          rpc_api.TicketStatus.COMPLETED,
          rpc_api.TicketStatus.FAILED,
      ):
        return status
      if timeout is not None and (time.time() - start) > timeout:
        raise TimeoutError(f"ticket {ticket_id} did not complete in {timeout}s")
      time.sleep(poll_interval)

  # non-blocking initiate methods (return ticket_id immediately)

  def initiate_trajectory_motion(
      self,
      trajectory_name: str,
      period_seconds: float | None = None,
      motion_type: rpc_api.TrajectoryMotionType = (
          rpc_api.TrajectoryMotionType.FULL
      ),
      static_gripper: bool = False,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate trajectory motion. Returns immediately with ticket_id.

    Args:
      trajectory_name: Name of the trajectory in the library.
      period_seconds: Optional duration override for execution.
      motion_type: How to execute the trajectory.
      static_gripper: Whether to keep the gripper static.
    """
    query = rpc_api.TrajectoryMotionQuery(
        trajectory_name=trajectory_name,
        period_seconds=period_seconds,
        motion_type=motion_type,
        static_gripper=static_gripper,
    )
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.trajectory_motion", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_visual_pose_motion(
      self,
      visual_pose_name: str,
      period_seconds: float,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate visual pose motion. Returns immediately with ticket_id.

    Args:
      visual_pose_name: Name of the visual pose to execute.
      period_seconds: Duration for the motion.
    """
    query = rpc_api.VisualPoseMotionQuery(
        visual_pose_name=visual_pose_name,
        period_seconds=period_seconds,
    )
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.visual_pose_motion", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_open_gripper(
      self,
      target_position: float | None = None,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate open gripper. Returns immediately with ticket_id.

    Args:
      target_position: Target gripper position, or None for default.
    """
    query = (
        rpc_api.OpenGripperQuery(target_position=target_position)
        if target_position is not None
        else rpc_api.OpenGripperQuery()
    )
    result = _rpc_call(self._get_rpc_client(), "behaviour.open_gripper", query)
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_close_gripper(
      self,
      target_position: float | None = None,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate close gripper. Returns immediately with ticket_id.

    Args:
      target_position: Target gripper position, or None for default.
    """
    query = (
        rpc_api.CloseGripperQuery(target_position=target_position)
        if target_position is not None
        else rpc_api.CloseGripperQuery()
    )
    result = _rpc_call(self._get_rpc_client(), "behaviour.close_gripper", query)
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_go_to_joints(
      self,
      configuration: np.ndarray | Sequence[float],
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate go to joints. Returns immediately with ticket_id.

    Args:
      configuration: Target joint configuration for the arm.
    """
    query = rpc_api.GoToJointsQuery(configuration=np.array(configuration))
    result = _rpc_call(self._get_rpc_client(), "behaviour.go_to_joints", query)
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_execute_learned_behavior(
      self,
      query: rpc_api.ExecuteLearnedBehaviorQuery,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiates a learned behavior. Returns immediately with ticket_id.

    Args:
      query: Query containing the behavior name.
    """
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.execute_learned_behavior", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_go_to_neutral_pose(
      self,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate go to neutral pose. Returns immediately with ticket_id."""
    query = rpc_api.GoToNeutralPoseQuery()
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.go_to_neutral_pose", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_align_leader_with_follower(
      self,
      timeout_seconds: float = 5.0,
      threshold: float = 0.1,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate align leader with follower. Returns immediately with ticket_id.

    Args:
      timeout_seconds: Maximum seconds to wait for alignment.
      threshold: Joint position threshold for alignment completion.
    """
    query = rpc_api.AlignLeaderWithFollowerQuery(
        timeout_seconds=timeout_seconds,
        threshold=threshold,
    )
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.align_leader_with_follower", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_wait_for_object(
      self,
      object_names: Sequence[str],
      timeout_seconds: float | None = None,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate wait for object. Returns immediately with ticket_id.

    Args:
      object_names: Names of objects to wait for (any match succeeds).
      timeout_seconds: Maximum seconds to wait for detection.
    """
    query = rpc_api.WaitForObjectQuery(
        object_names=list(object_names),
        timeout_seconds=timeout_seconds,
    )
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.wait_for_object", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  # non-blocking convenience methods that return futures

  def trajectory_motion(
      self,
      trajectory_name: str,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
      period_seconds: float | None = None,
      motion_type: rpc_api.TrajectoryMotionType = (
          rpc_api.TrajectoryMotionType.FULL
      ),
      static_gripper: bool = False,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue trajectory motion and return a future.

    Args:
      trajectory_name: Name of the trajectory in the library.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
      period_seconds: Optional duration override for execution.
      motion_type: How to execute the trajectory.
      static_gripper: Whether to keep the gripper static.
    """
    return self._submit_behaviour(
        lambda: self.initiate_trajectory_motion(
            trajectory_name=trajectory_name,
            period_seconds=period_seconds,
            motion_type=motion_type,
            static_gripper=static_gripper,
        ),
        timeout=timeout,
        arm=arm,
        behaviour_type="trajectory_motion",
    )

  def visual_pose_motion(
      self,
      visual_pose_name: str,
      period_seconds: float,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue visual pose motion and return a future.

    Args:
      visual_pose_name: Name of the visual pose to execute.
      period_seconds: Duration for the motion.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_visual_pose_motion(
            visual_pose_name=visual_pose_name,
            period_seconds=period_seconds,
        ),
        timeout=timeout,
        arm=arm,
        behaviour_type="visual_pose_motion",
    )

  def open_gripper(
      self,
      target_position: float | None = None,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue open gripper and return a future.

    Args:
      target_position: Target gripper position, or None for default.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_open_gripper(
            target_position=target_position,
        ),
        timeout=timeout,
        arm=arm,
        behaviour_type="open_gripper",
    )

  def close_gripper(
      self,
      target_position: float | None = None,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue close gripper and return a future.

    Args:
      target_position: Target gripper position, or None for default.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_close_gripper(
            target_position=target_position,
        ),
        timeout=timeout,
        arm=arm,
        behaviour_type="close_gripper",
    )

  def go_to_joints(
      self,
      configuration: np.ndarray | Sequence[float],
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue go to joints and return a future.

    Args:
      configuration: Target joint configuration for the arm.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_go_to_joints(
            configuration=configuration,
        ),
        timeout=timeout,
        arm=arm,
    )

  def execute_learned_behavior(
      self,
      query: rpc_api.ExecuteLearnedBehaviorQuery,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue execution of a learned behavior and return a future.

    Args:
      query: Query containing the behavior name.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_execute_learned_behavior(query),
        timeout=timeout,
        arm=arm,
        behaviour_type="execute_learned_behavior",
    )

  def go_to_neutral_pose(
      self,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue go to neutral pose and return a future.

    Args:
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        self.initiate_go_to_neutral_pose,
        timeout=timeout,
        arm=arm,
        behaviour_type="go_to_neutral_pose",
    )

  def align_leader_with_follower(
      self,
      timeout_seconds: float = 5.0,
      threshold: float = 0.1,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Align leader arm with follower and return a future.

    Args:
      timeout_seconds: Maximum seconds for alignment to complete.
      threshold: Joint position threshold for alignment.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_align_leader_with_follower(
            timeout_seconds=timeout_seconds,
            threshold=threshold,
        ),
        timeout=timeout,
        arm=arm,
        behaviour_type="align_leader_with_follower",
    )

  def wait_for_object(
      self,
      object_names: Sequence[str],
      timeout_seconds: float = 30.0,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue wait for object and return a future.

    Args:
      object_names: Names of objects to wait for (any match succeeds).
      timeout_seconds: Maximum seconds to wait for detection.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_wait_for_object(
            object_names=object_names,
            timeout_seconds=timeout_seconds,
        ),
        timeout=None,
        arm=arm,
        behaviour_type="wait_for_object",
    )

  # ticket status methods

  def get_ticket_status(self, ticket_id: str) -> rpc_api.TicketStatusResponse:
    """Get the current status of a behaviour ticket.

    Args:
      ticket_id: The ticket ID to query.
    """
    query = rpc_api.TicketStatusQuery(ticket_id=ticket_id)
    result = _rpc_call(self._get_rpc_client(), "behaviour.ticket_status", query)
    assert isinstance(result, rpc_api.TicketStatusResponse)
    return result

  def get_ticket_logs(
      self, ticket_id: str, since_index: int = 0
  ) -> rpc_api.TicketLogsResponse:
    """Get logs for a behaviour ticket.

    Args:
      ticket_id: The ticket ID to get logs for.
      since_index: Return logs starting from this index (for pagination).
    """
    query = rpc_api.TicketLogsQuery(
        ticket_id=ticket_id, since_index=since_index
    )
    result = _rpc_call(self._get_rpc_client(), "behaviour.ticket_logs", query)
    assert isinstance(result, rpc_api.TicketLogsResponse)
    return result

  def list_tickets(self) -> rpc_api.ListTicketsResponse:
    """List all behaviour tickets."""
    query = rpc_api.ListTicketsQuery()
    result = _rpc_call(self._get_rpc_client(), "behaviour.list_tickets", query)
    assert isinstance(result, rpc_api.ListTicketsResponse)
    return result

  def get_viewer_url(self) -> rpc_api.VisualisationUrlResponse:
    """Get the Rerun viewer URL for behaviour visualisation."""
    result = _rpc_call(self._get_rpc_client(), "behaviour.viewer_url")
    assert isinstance(result, rpc_api.VisualisationUrlResponse)
    return result


class ArmClient:
  """Arm-scoped client for behaviour/query calls.

  Wraps BehaviourClient methods to automatically specify the arm parameter.
  """

  def __init__(
      self,
      behaviour_client: BehaviourClient,
      arm: sdk_futures.ArmSide,
      query_client: QueryClient,
  ) -> None:
    """Initialize the arm client.

    Args:
      behaviour_client: The behaviour client to delegate to.
      arm: Which arm this client is scoped to.
      query_client: Query client for synchronous operations.
    """
    self._behaviour_client = behaviour_client
    self._arm = arm
    self._query_client = query_client

  def trajectory_motion(
      self,
      trajectory_name: str,
      timeout: float | None = None,
      period_seconds: float | None = None,
      motion_type: rpc_api.TrajectoryMotionType = (
          rpc_api.TrajectoryMotionType.FULL
      ),
      static_gripper: bool = False,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Execute a trajectory motion and return a future.

    Args:
      trajectory_name: Name of the trajectory in the library.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      period_seconds: Optional duration override for execution.
      motion_type: How to execute the trajectory.
      static_gripper: Whether to keep the gripper static.
    """
    return self._behaviour_client.trajectory_motion(
        trajectory_name=trajectory_name,
        timeout=timeout,
        arm=self._arm,
        period_seconds=period_seconds,
        motion_type=motion_type,
        static_gripper=static_gripper,
    )

  def visual_pose_motion(
      self,
      visual_pose_name: str,
      period_seconds: float,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Execute a visual pose motion and return a future.

    Args:
      visual_pose_name: Name of the visual pose in the library.
      period_seconds: Duration for the motion.
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.visual_pose_motion(
        visual_pose_name=visual_pose_name,
        period_seconds=period_seconds,
        timeout=timeout,
        arm=self._arm,
    )

  def open_gripper(
      self,
      target_position: float | None = None,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Open the gripper and return a future.

    Args:
      target_position: Target gripper position, or None for default.
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.open_gripper(
        target_position=target_position,
        timeout=timeout,
        arm=self._arm,
    )

  def close_gripper(
      self,
      target_position: float | None = None,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Close the gripper and return a future.

    Args:
      target_position: Target gripper position, or None for default.
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.close_gripper(
        target_position=target_position,
        timeout=timeout,
        arm=self._arm,
    )

  def go_to_neutral_pose(
      self,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Move the arm to a neutral pose and return a future.

    Args:
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.go_to_neutral_pose(
        timeout=timeout, arm=self._arm
    )

  def align_leader_with_follower(
      self,
      timeout_seconds: float = 5.0,
      threshold: float = 0.1,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Align leader arm with follower and return a future.

    Args:
      timeout_seconds: Maximum seconds for alignment to complete.
      threshold: Joint position threshold for alignment.
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.align_leader_with_follower(
        timeout_seconds=timeout_seconds,
        threshold=threshold,
        timeout=timeout,
        arm=self._arm,
    )

  def wait_for_object(
      self,
      object_names: Sequence[str],
      timeout_seconds: float = 30.0,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Wait until an object is detected and return a future.

    Args:
      object_names: Names of objects to wait for (any match succeeds).
      timeout_seconds: Maximum seconds to wait for detection.
    """
    return self._behaviour_client.wait_for_object(
        object_names=object_names,
        timeout_seconds=timeout_seconds,
        arm=self._arm,
    )

  def can_see_object(
      self,
      object_names: Sequence[str],
      timeout_seconds: float = 15.0,
  ) -> rpc_api.CanSeeObjectResponse:
    """Check if any of the specified objects are visible (blocking).

    Args:
      object_names: Names of objects to look for.
      timeout_seconds: Maximum seconds to wait for detection.
    """
    return self._query_client.can_see_object(
        object_names=object_names,
        timeout_seconds=timeout_seconds,
    )


class Robot:
  """Main client for robot control, aggregating all sub-clients.

  Provides access to execution mode control, raw sensor data, behaviour
  execution, trajectory/object/visual pose libraries, and recording.

  Example:
    robot = Robot("tcp://localhost:7532", "tcp://localhost:7533")
    robot.activate()
    robot.arm.open_gripper().result()
  """

  def __init__(
      self,
      server_address: str,
      query_server_address: str,
      use_compression: bool = False,
      timeout: int = 5000,
      query_timeout: int | None = None,
      primary_arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ):
    """Initialize the robot client.

    Args:
      server_address: Main RPC server address (e.g., "tcp://host:7532").
      query_server_address: Query server address (e.g., "tcp://host:7533").
      use_compression: Whether to compress RPC payloads with zstd.
      timeout: Default timeout in milliseconds for RPC calls.
      query_timeout: Timeout for query server, defaults to timeout if None.
      primary_arm: Arm selection exposed by the `arm` property.
    """
    base_client = client.BaseClient(
        server_address,
        use_compression=use_compression,
        timeout=timeout,
    )
    query_address = query_server_address
    query_client = client.BaseClient(
        query_address,
        use_compression=use_compression,
        timeout=timeout if query_timeout is None else query_timeout,
    )

    def _make_behaviour_client() -> client.BaseClient:
      return client.BaseClient(
          server_address,
          use_compression=use_compression,
          timeout=timeout,
      )

    self._exec_mode = ExecModeClient(base_client)
    self._raw_robot = RawRobotClient(base_client)
    self._query = QueryClient(query_client)
    self._recording = RecordingClient(base_client)
    self._episode_observer = EpisodeObserverClient(base_client)
    self._object_library = ObjectLibraryClient(base_client)
    self._trajectory_library = TrajectoryLibraryClient(base_client)
    self._visual_pose_library = VisualPoseLibraryClient(base_client)
    self._apriltag = AprilTagClient(base_client)
    self._behaviour = BehaviourClient(_make_behaviour_client)
    self._left_arm = ArmClient(
        self._behaviour,
        sdk_futures.ArmSide.LEFT,
        query_client=self._query,
    )
    self._right_arm = ArmClient(
        self._behaviour,
        sdk_futures.ArmSide.RIGHT,
        query_client=self._query,
    )
    if primary_arm == sdk_futures.ArmSide.LEFT:
      self._primary_arm = self._left_arm
    elif primary_arm == sdk_futures.ArmSide.RIGHT:
      self._primary_arm = self._right_arm
    else:
      raise ValueError(f"unsupported primary arm: {primary_arm}")

  @property
  def exec_mode(self) -> ExecModeClient:
    """Client for execution mode control."""
    return self._exec_mode

  @property
  def raw_robot(self) -> RawRobotClient:
    """Client for raw sensor data access."""
    return self._raw_robot

  @property
  def query(self) -> QueryClient:
    """Client for synchronous query operations."""
    return self._query

  @property
  def recording(self) -> RecordingClient:
    """Client for trajectory recording."""
    return self._recording

  @property
  def episode_observer(self) -> EpisodeObserverClient:
    """Client for episode recording observer (data gathering UI)."""
    return self._episode_observer

  @property
  def object_library(self) -> ObjectLibraryClient:
    """Client for object library management."""
    return self._object_library

  @property
  def trajectory_library(self) -> TrajectoryLibraryClient:
    """Client for trajectory library management."""
    return self._trajectory_library

  @property
  def visual_pose_library(self) -> VisualPoseLibraryClient:
    """Client for visual pose library management."""
    return self._visual_pose_library

  @property
  def apriltag(self) -> AprilTagClient:
    """Client for AprilTag detection services."""
    return self._apriltag

  @property
  def behaviour(self) -> BehaviourClient:
    """Client for behaviour execution."""
    return self._behaviour

  @property
  def arm(self) -> "ArmClient":
    """Primary arm client for single-arm usage."""
    return self._primary_arm

  @property
  def left_arm(self) -> "ArmClient":
    """Advanced access to the left arm client."""
    return self._left_arm

  @property
  def right_arm(self) -> "ArmClient":
    """Advanced access to the right arm client."""
    return self._right_arm

  def add_visual_pose_from_frame(
      self,
      name: str,
      description: str,
      reference_type: rpc_api.VisualReference,
      rgb_image: np.ndarray,
      reference_mask: np.ndarray,
      allow_overwrite: bool = False,
      depth_image: np.ndarray | None = None,
      camera: rpc_api.CameraType = rpc_api.CameraType.WRIST,
  ) -> rpc_api.AddVisualPoseQueryResponse:
    """Add a visual pose using an RGB frame and mask.

    If no depth image is provided, the method captures depth from the specified
    robot camera. When depth is unavailable, a zero-filled depth image is used.

    Args:
      name: Name of the visual pose.
      description: Human-readable description for the pose.
      reference_type: Type of visual reference (AR marker or object).
      rgb_image: RGB frame as [H, W, 3] uint8 array.
      reference_mask: Binary mask for the reference as [H, W] array.
      allow_overwrite: Whether to overwrite an existing pose with the same name.
      depth_image: Optional depth image as [H, W, 1] array.
      camera: Camera to query for depth if depth_image is None.

    Returns:
      Response indicating whether the pose was added.
    """
    resolved_depth = depth_image
    if resolved_depth is None:
      camera_response = self._raw_robot.get_camera_data(camera=camera)
      resolved_depth = camera_response.depth

    if resolved_depth is None:
      resolved_depth = np.zeros((*rgb_image.shape[:2], 1), dtype=np.int16)

    pose = rpc_api.VisualPoseEntry(
        name=name,
        description=description,
        reference_type=reference_type,
        rgb_image=rgb_image,
        depth_image=resolved_depth,
        reference_mask=reference_mask,
    )
    return self._visual_pose_library.add_entry(
        pose=pose,
        allow_overwrite=allow_overwrite,
    )

  def detect_apriltags_from_camera(
      self,
      camera: rpc_api.CameraType = rpc_api.CameraType.WRIST,
      families: Sequence[rpc_api.AprilTagFamily] | None = None,
      tag_size: float | None = None,
      timeout: int | None = None,
  ) -> AprilTagCameraDetection:
    """Capture a camera frame and run AprilTag detection on it.

    Args:
      camera: Camera to capture the frame from.
      families: Tag families to detect, or None to detect all families.
      tag_size: Tag size in meters for pose estimation.
      timeout: Optional RPC timeout in milliseconds for detection.

    Returns:
      Combined camera data and AprilTag detection response.
    """
    resolved_families = list(families) if families is not None else None
    camera_data = self._raw_robot.get_camera_data(camera=camera)
    if camera_data.rgb is None:
      detections = rpc_api.AprilTagDetectResponse(
          detections=[],
          error="camera frame unavailable",
      )
      return AprilTagCameraDetection(
          camera_data=camera_data,
          detections=detections,
      )

    detections = self._apriltag.detect(
        image=camera_data.rgb,
        families=resolved_families,
        intrinsics=camera_data.intrinsics,
        tag_size=tag_size,
        timeout=timeout,
    )
    return AprilTagCameraDetection(
        camera_data=camera_data,
        detections=detections,
    )

  def activate(self) -> rpc_api.ExecutionModeQueryResponse:
    """Set the robot to READY mode for accepting behaviour commands.

    Raises:
      RuntimeError: If the mode transition fails.
    """
    response = self._exec_mode.set_execution_mode(
        new_mode=rpc_api.ExecutionMode.READY
    )
    confirmed = self._exec_mode.get_execution_mode()
    if confirmed.current_mode != rpc_api.ExecutionMode.READY:
      raise RuntimeError(
          "failed to set execution mode to READY "
          f"(got {confirmed.current_mode})"
      )
    return response

  def deactivate(self) -> rpc_api.ExecutionModeQueryResponse:
    """Set the robot to STOP mode, parking the arm at zero position.

    Raises:
      RuntimeError: If the mode transition fails.
    """
    response = self._exec_mode.set_execution_mode(
        new_mode=rpc_api.ExecutionMode.STOP
    )
    confirmed = self._exec_mode.get_execution_mode()
    if confirmed.current_mode != rpc_api.ExecutionMode.STOP:
      raise RuntimeError(
          f"failed to set execution mode to STOP (got {confirmed.current_mode})"
      )
    return response
