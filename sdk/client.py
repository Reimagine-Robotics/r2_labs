"""High-level clients for robot control and behaviour execution."""

import dataclasses
import functools
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
      Response containing camera availability, RGB/depth frames, and intrinsics.
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

  def get_button_peripherals(
      self,
  ) -> rpc_api.ButtonPeripheralQueryResponse:
    """Get raw button states for cuff and pedal input sources."""
    result = _rpc_call(self._rpc_client, "raw_robot.get_button_peripherals")
    assert isinstance(result, rpc_api.ButtonPeripheralQueryResponse)
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

  def predict_progress(
      self,
      model_id: str = "",
      service_address: str = "",
  ) -> rpc_api.PredictProgressResponse:
    """Predict task completion progress from current camera image.

    Args:
      model_id: Model ID for local progress prediction.
      service_address: Service address for remote inference
        (e.g. tcp://gpu-machine:4244).

    Returns:
      Response containing predicted progress value in [0, 1].

    Raises:
      ValueError: If both or neither of model_id and service_address are set.
    """
    query = rpc_api.PredictProgressQuery(
        model_id=model_id,
        service_address=service_address,
    )
    result = _rpc_call(
        self._rpc_client,
        "query.predict_progress",
        query,
    )
    assert isinstance(result, rpc_api.PredictProgressResponse)
    return result


class RecordingClient:
  """Client for trajectory recording operations.

  Usage:
    # 1. Prepare for recording (sets trajectory type and execution mode)
    robot.recording.prepare()

    # 2. Start recording (or press recording-toggle control)
    robot.recording.start()

    # 3. Stop recording and get trajectory (or press recording-toggle control,
    #    or wait
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
    the recording-toggle control (for example cuff button D or pedal A).

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


class VisualRecordingClient:
  """Client for visual trajectory recording operations.

  Usage:
    # 1. Prepare for recording (sets execution mode)
    robot.visual_trajectory_recording.prepare()

    # 2. Start recording (or press recording-toggle control)
    robot.visual_trajectory_recording.start()

    # 3. Stop recording (or press recording-toggle control, or wait for timeout)
    response = robot.visual_trajectory_recording.stop()

    # 4. Fetch individual frames for annotation
    frame = robot.visual_trajectory_recording.get_frame(frame_index=0)

    # 5. Save with reference masks
    robot.visual_trajectory_recording.save(name="pick", reference_masks=masks, ...)
  """

  def __init__(self, rpc_client: client.BaseClient) -> None:
    self._rpc_client = rpc_client

  def prepare(
      self,
      trajectory_source: rpc_api.TrajectorySource = (
          rpc_api.TrajectorySource.ROBOT
      ),
      timeout_seconds: float | None = 300.0,
      hold_until_start: bool = False,
  ) -> rpc_api.PrepareVisualRecordingResponse:
    """Prepare for visual recording.

    This clears any previously recorded trajectory. The robot will be switched
    to the specified execution mode based on trajectory source (TEACH or TELEOP).

    Args:
      trajectory_source: Source of the trajectory data (ROBOT or TELEOP).
      timeout_seconds: Auto-stop after duration, or None to disable.
      hold_until_start: If True, defer execution ∏mode change until start() is called.

    Returns:
      Response with error field set if preparation failed.
    """
    query = rpc_api.PrepareVisualRecordingQuery(
        trajectory_source=trajectory_source,
        timeout_seconds=timeout_seconds,
        hold_until_start=hold_until_start,
    )
    result = _rpc_call(self._rpc_client, "visual_recording.prepare", query)
    assert isinstance(result, rpc_api.PrepareVisualRecordingResponse)
    return result

  def start(self) -> rpc_api.StartVisualRecordingResponse:
    """Start recording samples. Must call prepare() first."""
    result = _rpc_call(self._rpc_client, "visual_recording.start")
    assert isinstance(result, rpc_api.StartVisualRecordingResponse)
    return result

  def stop(self) -> rpc_api.StopVisualRecordingResponse:
    """Stop recording and return frame count + period.

    Idempotent: returns cached result if already stopped. Data stays on
    the server until the next prepare() call.
    """
    result = _rpc_call(self._rpc_client, "visual_recording.stop")
    assert isinstance(result, rpc_api.StopVisualRecordingResponse)
    return result

  def get_state(self) -> rpc_api.VisualRecordingStateResponse:
    """Get the current visual recording state."""
    result = _rpc_call(self._rpc_client, "visual_recording.get_state")
    assert isinstance(result, rpc_api.VisualRecordingStateResponse)
    return result

  def get_frame(
      self, frame_index: int
  ) -> rpc_api.GetVisualRecordingFrameResponse:
    """Get a single recorded frame by index.

    Args:
      frame_index: Zero-based index of the frame to fetch.

    Returns:
      Response with rgb and depth arrays, or None if index out of range.
    """
    query = rpc_api.GetVisualRecordingFrameQuery(frame_index=frame_index)
    result = _rpc_call(self._rpc_client, "visual_recording.get_frame", query)
    assert isinstance(result, rpc_api.GetVisualRecordingFrameResponse)
    return result

  def save(
      self,
      name: str,
      description: str = "",
      reference_type: rpc_api.VisualReference = rpc_api.VisualReference.OBJECT,
      camera_type: rpc_api.CameraType = rpc_api.CameraType.WRIST,
      reference_masks: np.ndarray | None = None,
      apriltag_metadata: rpc_api.AprilTagPoseMetadata | None = None,
      allow_overwrite: bool = False,
      start_frame: int | None = None,
      end_frame: int | None = None,
  ) -> rpc_api.SaveVisualRecordingResponse:
    """Save the recorded visual trajectory with reference masks.

    Combines server-side recorded data with client-provided masks.

    Args:
      name: Name for the visual trajectory entry.
      description: Optional description.
      reference_type: OBJECT or APRILTAG.
      camera_type: Camera type used for recording.
      reference_masks: Boolean masks [T, H, W], one per frame.
      apriltag_metadata: Required if reference_type is APRILTAG.
      allow_overwrite: If True, overwrite existing entry with same name.
      start_frame: If set, trim recording to start at this frame (inclusive).
      end_frame: If set, trim recording to end at this frame (inclusive).

    Returns:
      Response with error field set if save failed.
    """
    query = rpc_api.SaveVisualRecordingQuery(
        name=name,
        description=description,
        reference_type=reference_type,
        camera_type=camera_type,
        reference_masks=reference_masks,
        apriltag_metadata=apriltag_metadata,
        allow_overwrite=allow_overwrite,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    result = _rpc_call(self._rpc_client, "visual_recording.save", query)
    assert isinstance(result, rpc_api.SaveVisualRecordingResponse)
    return result

  def load_from_saved(
      self, name: str
  ) -> rpc_api.LoadVisualTrajectoryIntoBufferResponse:
    """Load a saved trajectory's frames into the recording buffer.

    After loading, get_frame_thumbnails(), segment_recording(), etc. work
    as if the data was freshly recorded.

    Args:
      name: Name of the saved visual trajectory to load.
    """
    query = rpc_api.LoadVisualTrajectoryIntoBufferQuery(name=name)
    result = _rpc_call(
        self._rpc_client, "visual_recording.load_from_saved", query
    )
    assert isinstance(result, rpc_api.LoadVisualTrajectoryIntoBufferResponse)
    return result

  def get_frame_thumbnails(
      self,
  ) -> rpc_api.GetVisualRecordingFrameThumbnailsResponse:
    """Get small thumbnail images for all recorded frames."""
    result = _rpc_call(
        self._rpc_client, "visual_recording.get_frame_thumbnails"
    )
    assert isinstance(result, rpc_api.GetVisualRecordingFrameThumbnailsResponse)
    return result

  def segment_recording(
      self,
      positive_points: list[np.ndarray],
      negative_points: list[np.ndarray],
      subsample: int = rpc_api.DEFAULT_ANNOTATION_SUBSAMPLE,
      start_frame: int | None = None,
      end_frame: int | None = None,
      mode: rpc_api.SegmentationMode = "sam2",
      timeout: int = 180000,
  ) -> rpc_api.SegmentVisualRecordingResponse:
    """Run segmentation on the recorded frames (server-side).

    Frames are read directly from the server's recording buffer.

    Args:
      positive_points: Points on the object as list of [N, 3] int32 (T, Y, X).
      negative_points: Points not on the object as list of [N, 3] int32 (T, Y, X).
      subsample: Keep every Nth frame for segmentation. 1 = all frames.
      start_frame: Optional first frame index to process (inclusive).
      end_frame: Optional last frame index to process (inclusive).
      mode: Segmentation mode — "sam2" or "depth".
      timeout: RPC timeout in milliseconds.

    Returns:
      Response with segmentation_mask [T, H, W] bool array.
    """
    query = rpc_api.SegmentVisualRecordingQuery(
        positive_points=positive_points,
        negative_points=negative_points,
        subsample=subsample,
        start_frame=start_frame,
        end_frame=end_frame,
        mode=mode,
    )
    result = _rpc_call(
        self._rpc_client,
        "visual_recording.segment_recording",
        query,
        timeout=timeout,
    )
    assert isinstance(result, rpc_api.SegmentVisualRecordingResponse)
    return result

  def generate_apriltag_masks(
      self,
      tag_family: rpc_api.AprilTagFamily,
      tag_id: int,
      tag_size: float,
      start_frame: int | None = None,
      end_frame: int | None = None,
      timeout: int = 60000,
  ) -> rpc_api.GenerateAprilTagMasksResponse:
    """Detect an AprilTag across all recorded frames and generate masks.

    Args:
      tag_family: The AprilTag family to detect.
      tag_id: The specific tag ID to track.
      tag_size: Physical tag size in meters.
      start_frame: Optional first frame index to process (inclusive).
      end_frame: Optional last frame index to process (inclusive).
      timeout: RPC timeout in milliseconds.

    Returns:
      Response with segmentation_mask [T, H, W] bool and apriltag_metadata.
    """
    query = rpc_api.GenerateAprilTagMasksQuery(
        tag_family=tag_family,
        tag_id=tag_id,
        tag_size=tag_size,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    result = _rpc_call(
        self._rpc_client,
        "visual_recording.generate_apriltag_masks",
        query,
        timeout=timeout,
    )
    assert isinstance(result, rpc_api.GenerateAprilTagMasksResponse)
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

  def save(self, entry_prefix: str) -> None:
    """Save the current episode.

    Args:
      entry_prefix: Optional entry prefix for saving the episode.
    """
    query = rpc_api.EpisodeObserverSaveQuery(entry_prefix=entry_prefix)
    _rpc_call(self._rpc_client, "episode_observer.save", query, timeout=30_000)

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

  def set_is_human(self, is_human: bool) -> None:
    """Set the is_human flag for subsequent timesteps.

    When is_human=True, the episode observer will inject is_human=True into
    observations for all timesteps recorded until is_human is set to False.
    This is used for DAgger-style data collection where human interventions
    need to be tracked per-timestep.

    Args:
      is_human: Whether the current control is human (True) or policy (False).
    """
    query = rpc_api.SetIsHumanQuery(is_human=is_human)
    _rpc_call(self._rpc_client, "episode_observer.set_is_human", query)


class CollectDataClient:
  """Client for backend-owned collect-data workflow orchestration."""

  def __init__(self, rpc_client: client.BaseClient) -> None:
    self._rpc_client = rpc_client

  def prepare(
      self,
      *,
      continuous_teleop: bool | None = None,
      start_trajectory: str | None = None,
      align_timeout_seconds: float | None = None,
      align_threshold: float | None = None,
      behaviour_wait_timeout_seconds: float | None = None,
  ) -> rpc_api.CollectDataPrepareResponse:
    query = rpc_api.CollectDataPrepareQuery(
        continuous_teleop=continuous_teleop,
        start_trajectory=start_trajectory,
        align_timeout_seconds=align_timeout_seconds,
        align_threshold=align_threshold,
        behaviour_wait_timeout_seconds=behaviour_wait_timeout_seconds,
    )
    result = _rpc_call(self._rpc_client, "collect_data.prepare", query)
    assert isinstance(result, rpc_api.CollectDataPrepareResponse)
    return result

  def start(self) -> rpc_api.CollectDataStartResponse:
    result = _rpc_call(self._rpc_client, "collect_data.start")
    assert isinstance(result, rpc_api.CollectDataStartResponse)
    return result

  def stop(self) -> rpc_api.CollectDataStopResponse:
    result = _rpc_call(self._rpc_client, "collect_data.stop")
    assert isinstance(result, rpc_api.CollectDataStopResponse)
    return result

  def save(self, entry_prefix: str) -> rpc_api.CollectDataSaveResponse:
    query = rpc_api.CollectDataSaveQuery(entry_prefix=entry_prefix)
    result = _rpc_call(
        self._rpc_client, "collect_data.save", query, timeout=30_000
    )
    assert isinstance(result, rpc_api.CollectDataSaveResponse)
    return result

  def discard(self) -> rpc_api.CollectDataDiscardResponse:
    result = _rpc_call(self._rpc_client, "collect_data.discard")
    assert isinstance(result, rpc_api.CollectDataDiscardResponse)
    return result

  def get_state(self) -> rpc_api.CollectDataStateResponse:
    result = _rpc_call(self._rpc_client, "collect_data.get_state")
    assert isinstance(result, rpc_api.CollectDataStateResponse)
    return result

  def set_task_description(self, description: str) -> None:
    query = rpc_api.SetTaskDescriptionQuery(description=description)
    _rpc_call(self._rpc_client, "collect_data.set_task_description", query)

  def set_is_human(self, is_human: bool) -> None:
    query = rpc_api.SetIsHumanQuery(is_human=is_human)
    _rpc_call(self._rpc_client, "collect_data.set_is_human", query)


class DaggerClient:
  """Client for DAgger policy-assist orchestration."""

  def __init__(self, rpc_client: client.BaseClient) -> None:
    self._rpc_client = rpc_client

  def configure(
      self, query: rpc_api.DaggerConfigQuery
  ) -> rpc_api.DaggerConfigureResponse:
    result = _rpc_call(self._rpc_client, "dagger.configure", query)
    assert isinstance(result, rpc_api.DaggerConfigureResponse)
    return result

  def toggle(self) -> rpc_api.DaggerToggleResponse:
    result = _rpc_call(self._rpc_client, "dagger.toggle")
    assert isinstance(result, rpc_api.DaggerToggleResponse)
    return result

  def stop(self) -> rpc_api.DaggerStopResponse:
    result = _rpc_call(self._rpc_client, "dagger.stop")
    assert isinstance(result, rpc_api.DaggerStopResponse)
    return result

  def get_state(self) -> rpc_api.DaggerStateResponse:
    result = _rpc_call(self._rpc_client, "dagger.get_state")
    assert isinstance(result, rpc_api.DaggerStateResponse)
    return result


class EvalClient:
  """Client for blinded model evaluation orchestration."""

  def __init__(self, rpc_client: client.BaseClient) -> None:
    self._rpc_client = rpc_client

  def configure(
      self, query: rpc_api.EvalConfigQuery
  ) -> rpc_api.EvalConfigureResponse:
    result = _rpc_call(self._rpc_client, "eval.configure", query)
    assert isinstance(result, rpc_api.EvalConfigureResponse)
    return result

  def start(self) -> rpc_api.EvalStartResponse:
    result = _rpc_call(self._rpc_client, "eval.start")
    assert isinstance(result, rpc_api.EvalStartResponse)
    return result

  def advance_trial(self) -> rpc_api.EvalAdvanceResponse:
    result = _rpc_call(self._rpc_client, "eval.advance_trial")
    assert isinstance(result, rpc_api.EvalAdvanceResponse)
    return result

  def record_outcome(
      self, query: rpc_api.EvalRecordOutcomeQuery
  ) -> rpc_api.EvalRecordOutcomeResponse:
    result = _rpc_call(self._rpc_client, "eval.record_outcome", query)
    assert isinstance(result, rpc_api.EvalRecordOutcomeResponse)
    return result

  def stop_trial_policy(self) -> rpc_api.EvalStopTrialPolicyResponse:
    result = _rpc_call(self._rpc_client, "eval.stop_trial_policy")
    assert isinstance(result, rpc_api.EvalStopTrialPolicyResponse)
    return result

  def enable_teleop(self) -> rpc_api.EvalEnableTeleopResponse:
    result = _rpc_call(self._rpc_client, "eval.enable_teleop")
    assert isinstance(result, rpc_api.EvalEnableTeleopResponse)
    return result

  def stop(self) -> rpc_api.EvalStopResponse:
    result = _rpc_call(self._rpc_client, "eval.stop")
    assert isinstance(result, rpc_api.EvalStopResponse)
    return result

  def discard(self) -> rpc_api.EvalStopResponse:
    result = _rpc_call(self._rpc_client, "eval.discard")
    assert isinstance(result, rpc_api.EvalStopResponse)
    return result

  def upload(self) -> rpc_api.EvalUploadResponse:
    result = _rpc_call(self._rpc_client, "eval.upload")
    assert isinstance(result, rpc_api.EvalUploadResponse)
    return result

  def get_state(self) -> rpc_api.EvalStateResponse:
    result = _rpc_call(self._rpc_client, "eval.get_state")
    assert isinstance(result, rpc_api.EvalStateResponse)
    return result


class HardwareHealthClient:
  """Client for hardware health status."""

  def __init__(self, rpc_client: client.BaseClient) -> None:
    self._rpc_client = rpc_client

  def get_status(self) -> rpc_api.HardwareHealthResponse:
    result = _rpc_call(self._rpc_client, "hardware_health.get_status")
    assert isinstance(result, rpc_api.HardwareHealthResponse)
    return result


class ModelServicesClient:
  """Client for managing model inference services.

  Allows pre-loading models as services to eliminate load/warmup time when
  switching between different skill models.

  Example:
    # Start services for models
    address = robot.model_services.start("DCAM#tender-engineer-160")

    # Wait for service to become healthy
    robot.model_services.wait_until_ready(timeout=60)

    # Get all running services with updated health status
    services = robot.model_services.get_all()
    for svc in services:
        print(f"{svc.model_id} -> {svc.address} (healthy={svc.healthy})")

    # Stop all services to free GPU memory
    robot.model_services.stop_all()
  """

  def __init__(self, rpc_client: client.BaseClient) -> None:
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication. This should be a client
        connected to the training server since model services are managed by
        the training server to colocate with a GPU. Using a client connected to
        the main robot server will not work.
    """
    self._rpc_client = rpc_client

  def start(self, model_id: str, port: int | None = None) -> str:
    """Start an inference service for a model.

    Args:
      model_id: The model warehouse model ID to serve.
      port: Optional port to use. If None, a port is auto-assigned.

    Returns:
      The service address (e.g., "tcp://localhost:4601").
    """
    query = rpc_api.StartModelServiceQuery(model_id=model_id, port=port)
    result = _rpc_call(self._rpc_client, "model_services.start", query)
    assert isinstance(result, rpc_api.StartModelServiceResponse)
    return result.address

  def stop(self, model_id: str) -> None:
    """Stop an inference service.

    Args:
      model_id: The model ID of the service to stop.
    """
    query = rpc_api.StopModelServiceQuery(model_id=model_id)
    _rpc_call(self._rpc_client, "model_services.stop", query)

  def stop_all(self) -> None:
    """Stop all managed inference services."""
    _rpc_call(self._rpc_client, "model_services.stop_all")

  def get_all(self) -> list[rpc_api.ModelServiceInfo]:
    """Get all running inference services.

    Note: The 'healthy' flag in returned service info represents the last
    known health state, not real-time status. If you need current health
    information, call wait_until_ready() first to update the health status.

    Returns:
      List of service info objects with cached health status.
    """
    result = _rpc_call(self._rpc_client, "model_services.list")
    assert isinstance(result, rpc_api.ListModelServicesResponse)
    return result.services

  def wait_until_ready(
      self,
      model_ids: list[str] | None = None,
      timeout: float = 120.0,
      poll_interval: float = 1.0,
  ) -> rpc_api.WaitModelServicesResponse:
    """Wait for model services to become ready.

    Args:
      model_ids: List of model IDs to wait for. None = all services.
      timeout: Maximum seconds to wait.
      poll_interval: Seconds between health checks.

    Returns:
      WaitModelServicesResponse with success flag and lists of ready/pending
      models.
    """
    query = rpc_api.WaitModelServicesQuery(
        model_ids=model_ids,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    result = _rpc_call(
        self._rpc_client,
        "model_services.wait",
        query,
        timeout=int(timeout * 1000),
    )
    assert isinstance(result, rpc_api.WaitModelServicesResponse)
    return result

  def get_address(self, model_id: str) -> str | None:
    """Get the service address for a model.

    Args:
      model_id: The model ID to look up.

    Returns:
      Service address if running, None otherwise.
    """
    for svc in self.get_all():
      if svc.model_id == model_id:
        return svc.address
    return None


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


class VisualTrajectoryLibraryClient:
  """Client for managing visual trajectories."""

  def __init__(self, rpc_client: client.BaseClient):
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def list_entries(self) -> rpc_api.ListVisualTrajectoriesResponse:
    """List all visual trajectories in the library."""
    result = _rpc_call(
        self._rpc_client, "visual_trajectory_library.list_entries"
    )
    assert isinstance(result, rpc_api.ListVisualTrajectoriesResponse)
    return result

  def add_entry(
      self,
      visual_trajectory: rpc_api.VisualTrajectoryLibraryEntry,
      allow_overwrite: bool = False,
  ) -> rpc_api.AddVisualTrajectoryQueryResponse:
    """Add a visual trajectory to the library.

    Args:
      visual_trajectory: Visual trajectory entry to add.
      allow_overwrite: Whether to overwrite existing entry with same name.
    """
    entry = rpc_api.AddVisualTrajectoryQuery(
        visual_trajectory=visual_trajectory,
        allow_overwrite=allow_overwrite,
    )
    result = _rpc_call(
        self._rpc_client, "visual_trajectory_library.add_entry", entry
    )
    assert isinstance(result, rpc_api.AddVisualTrajectoryQueryResponse)
    return result

  def delete_entry(
      self, visual_trajectory_name: str
  ) -> rpc_api.DeleteVisualTrajectoryQueryResponse:
    """Delete a visual trajectory from the library.

    Args:
      visual_trajectory_name: Name of the visual trajectory to delete.
    """
    entry = rpc_api.DeleteVisualTrajectoryQuery(
        visual_trajectory_name=visual_trajectory_name
    )
    result = _rpc_call(
        self._rpc_client, "visual_trajectory_library.delete_entry", entry
    )
    assert isinstance(result, rpc_api.DeleteVisualTrajectoryQueryResponse)
    return result

  def load_entry(
      self, visual_trajectory_name: str
  ) -> rpc_api.LoadVisualTrajectoryQueryResponse:
    """Load a visual trajectory from the library by name.

    Args:
      visual_trajectory_name: Name of the visual trajectory to load.
    """
    query = rpc_api.LoadVisualTrajectoryQuery(
        visual_trajectory_name=visual_trajectory_name
    )
    result = _rpc_call(
        self._rpc_client, "visual_trajectory_library.load_entry", query
    )
    assert isinstance(result, rpc_api.LoadVisualTrajectoryQueryResponse)
    return result

  def update_masks(
      self,
      name: str,
      reference_masks: np.ndarray,
      reference_type: rpc_api.VisualReference,
      apriltag_metadata: rpc_api.AprilTagPoseMetadata | None = None,
  ) -> rpc_api.UpdateVisualTrajectoryMasksResponse:
    """Update masks and reference type on an existing trajectory.

    Args:
      name: Name of the visual trajectory to update.
      reference_masks: New reference masks [T, H, W] bool array.
      reference_type: New reference type.
      apriltag_metadata: AprilTag metadata if reference_type is APRILTAG.
    """
    query = rpc_api.UpdateVisualTrajectoryMasksQuery(
        name=name,
        reference_masks=reference_masks,
        reference_type=reference_type,
        apriltag_metadata=apriltag_metadata,
    )
    result = _rpc_call(
        self._rpc_client,
        "visual_trajectory_library.update_masks",
        query,
    )
    assert isinstance(result, rpc_api.UpdateVisualTrajectoryMasksResponse)
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

  def initiate_visual_trajectory_motion(
      self,
      visual_trajectory_name: str,
      static_gripper: bool = False,
      motion_type: rpc_api.TrajectoryMotionType = rpc_api.TrajectoryMotionType.FULL,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate visual trajectory motion. Returns immediately with ticket_id.

    Args:
      visual_trajectory_name: Name of the visual trajectory to execute.
      static_gripper: Whether to keep the gripper static.
      motion_type: FULL plays the entire trajectory. GO_TO_START uses visual
        servoing to move to the first frame. GO_TO_END is not supported.
    """
    query = rpc_api.VisualTrajectoryMotionQuery(
        visual_trajectory_name=visual_trajectory_name,
        motion_type=motion_type,
        static_gripper=static_gripper,
    )
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.visual_trajectory_motion", query
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

  def visual_trajectory_motion(
      self,
      visual_trajectory_name: str,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
      static_gripper: bool = False,
      motion_type: rpc_api.TrajectoryMotionType = rpc_api.TrajectoryMotionType.FULL,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue visual trajectory motion and return a future.

    Args:
      visual_trajectory_name: Name of the visual trajectory to execute.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
      static_gripper: Whether to keep the gripper static.
      motion_type: FULL plays the entire trajectory. GO_TO_START uses visual
        servoing to move to the first frame. GO_TO_END is not supported.
    """
    return self._submit_behaviour(
        lambda: self.initiate_visual_trajectory_motion(
            visual_trajectory_name=visual_trajectory_name,
            static_gripper=static_gripper,
            motion_type=motion_type,
        ),
        timeout=timeout,
        arm=arm,
        behaviour_type="visual_trajectory_motion",
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

  def get_replay_notebook_cells(
      self, ticket_ids: Sequence[str]
  ) -> rpc_api.ReplayNotebookCellsResponse:
    """Build notebook replay cells for one or more behaviour tickets.

    Args:
      ticket_ids: Ticket IDs to translate into notebook cell source.
    """
    query = rpc_api.ReplayNotebookCellsQuery(ticket_ids=list(ticket_ids))
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.replay_notebook_cells", query
    )
    assert isinstance(result, rpc_api.ReplayNotebookCellsResponse)
    return result

  def get_viewer_url(self) -> rpc_api.VisualisationUrlResponse:
    """Get the Rerun viewer URL for behaviour visualisation."""
    result = _rpc_call(self._get_rpc_client(), "behaviour.viewer_url")
    assert isinstance(result, rpc_api.VisualisationUrlResponse)
    return result


class TrainerClient:
  """Client for training models.

  Usage:
    # Start training with automatic dataset building
    response = robot.trainer.train_skill_model(
        model_name="pick_up_can_model",
        entry_filter="pick_up_can*",  # Matches entries in data warehouse
        training_steps=50000,
        batch_size=64,
        prediction_horizon=32,
    )
    if response.error:
        print(f"Failed to start training: {response.error}")
    else:
        print(f"Dataset entries: {response.current_entry_count}")
        if response.dataset_was_rebuilt:
            print("Dataset was rebuilt from data warehouse")

  Monitoring:
    # Poll for training completion
    while True:
        status = robot.trainer.get_training_status()
        if status.is_finished:
            break
        print(f"Training: {status.steps_completed} steps, loss={status.loss}")
        time.sleep(10.0)

    # To cancel training early:
    # robot.trainer.cancel_training()
  """

  def __init__(self, rpc_client: client.BaseClient) -> None:
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def train_skill_model(
      self,
      model_name: str,
      training_steps: int,
      entry_filters: list[str],
      entry_tags: list[str] | None = None,
      cameras: list[str] | None = None,
      model_save_dir: str = "",
      force_rebuild: bool = False,
      batch_size: int = 64,
      prediction_horizon: int = 32,
      use_joint_torques: bool = False,
      checkpoint_interval_steps: int = 1000,
      max_checkpoints_to_keep: int = 10,
      timeout: int | None = None,
  ) -> rpc_api.StartSkillTrainingResponse:
    """Start model training for a robot skill.

    Initiates asynchronous model training on the server. Only one training
    run can be active at a time. Use get_training_status() to monitor
    progress and cancel_training() to stop early.

    Args:
      model_name: Name for the exported model in the model warehouse.
      training_steps: Total number of training steps to run.
      entry_filters: List of glob patterns for selecting data warehouse entries
          (e.g., ["pick_up_can*", "open_door*"]). Automatically builds and
          caches the dataset. Multiple patterns are combined.
      entry_tags: Required data warehouse tags for entry filtering.
      cameras: Camera names. None uses default cameras; empty list means
          no cameras.
      model_save_dir: Optional directory to save checkpoints. If empty,
          uses default location.
      force_rebuild: If True, rebuild the dataset even if cached version exists.
      batch_size: Training batch size.
      prediction_horizon: Number of future timesteps to predict.
      use_joint_torques: Whether to include piper_joint_torques in proprio.
      checkpoint_interval_steps: Save checkpoint every N steps. Default 1000.
      max_checkpoints_to_keep: Max checkpoints to keep. Default 10.
      timeout: Optional RPC timeout in milliseconds.

    Returns:
      Response containing:
        - error: Error message if training could not be started, None on success.

      Use get_training_status() to monitor progress. The status includes:
        - phase: Current phase ("idle", "preparing_dataset", "training", "finished", "failed")
        - export_entries_processed / export_entries_total: Dataset export progress

    Raises:
      ValueError: If entry_filters is empty.
      RuntimeError: If training is already running on the server.
    """
    if not entry_filters:
      raise ValueError("entry_filters must be a non-empty list")

    # Check if training is already running
    if self.is_training_running():
      status = self.get_training_status()
      raise RuntimeError(
          f"Training is already running on the server "
          f"(step {status.steps_completed}/{status.max_steps}, loss={status.loss:.4f}). "
          f"Please either wait for it to finish or call cancel_training() first."
      )

    query = rpc_api.StartSkillTrainingQuery(
        model_name=model_name,
        training_steps=training_steps,
        entry_filters=entry_filters,
        entry_tags=entry_tags or [],
        cameras=cameras,
        model_save_dir=model_save_dir,
        force_rebuild=force_rebuild,
        batch_size=batch_size,
        prediction_horizon=prediction_horizon,
        use_joint_torques=use_joint_torques,
        checkpoint_interval_steps=checkpoint_interval_steps,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
    )
    result = _rpc_call(
        self._rpc_client,
        "trainer.train_skill_model",
        query,
        timeout=timeout,
    )
    assert isinstance(result, rpc_api.StartSkillTrainingResponse)
    return result

  def is_training_running(self) -> bool:
    """Check if training is currently running on the server.

    Returns:
      True if training is in progress, False otherwise.
    """
    status = self.get_training_status()
    # Not running if finished OR failed
    if status.is_finished or status.phase in ("finished", "failed", "idle"):
      return False
    return True

  def get_training_status(self) -> rpc_api.TrainingStatusResponse:
    """Get information about the current skill model training status.

    Returns:
      Response containing:
        - is_finished: Whether training has completed
        - steps_completed: Current training step
        - max_steps: Total steps configured
        - loss: Current training loss
        - fps: Training speed (steps per second)
        - seconds_per_step: Time per training step
        - metrics: Additional training metrics dict
        - phase: Current phase ("idle", "preparing_dataset", "training", "finished", "failed")
        - export_entries_processed: Number of entries exported so far
        - export_entries_total: Total entries to export
    """
    result = _rpc_call(self._rpc_client, "trainer.get_training_status")
    assert isinstance(result, rpc_api.TrainingStatusResponse)
    return result

  def cancel_training(self) -> rpc_api.CancelTrainingResponse:
    """Cancel the current skill model training.

    Saves a checkpoint before stopping.

    Returns:
      Response containing:
        - success: Whether cancellation was successful
        - error: Error message if cancellation failed
    """
    query = rpc_api.CancelTrainingQuery()
    result = _rpc_call(self._rpc_client, "trainer.cancel_training", query)
    assert isinstance(result, rpc_api.CancelTrainingResponse)
    return result

  def reset_trainer(self) -> rpc_api.ResetTrainerResponse:
    """Reset the trainer to clean slate - cancel training and clear all state.

    This stops any running training and resets the trainer to initial idle state.

    Returns:
      Response containing:
        - success: Whether reset was successful
        - error: Error message if reset failed
    """
    result = _rpc_call(self._rpc_client, "trainer.reset_trainer")
    assert isinstance(result, rpc_api.ResetTrainerResponse)
    return result

  def list_models(self) -> list[dict[str, Any]]:
    """List all exported models from the model warehouse.

    Returns:
      List of model dicts with model_id, timestamp, description, tags.
    """
    result = _rpc_call(self._rpc_client, "trainer.list_models")
    assert isinstance(result, list)
    return result

  def list_model_names_from_checkpoints(self) -> list[str]:
    """List model names that have saved checkpoints.

    Returns:
      List of model names (from checkpoint directory names).
    """
    result = _rpc_call(
        self._rpc_client, "trainer.list_model_names_from_checkpoints"
    )
    assert isinstance(result, list)
    return result

  def list_entry_filters(
      self, search: str = ""
  ) -> rpc_api.ListEntryFiltersResponse:
    """List entry filter IDs from the data warehouse.

    Args:
      search: Optional search term to filter results.

    Returns:
      Response with list of unique entry filter IDs.
    """
    query = rpc_api.ListEntryFiltersQuery(search=search)
    result = _rpc_call(self._rpc_client, "trainer.list_entry_filters", query)
    assert isinstance(result, rpc_api.ListEntryFiltersResponse)
    return result

  def start_export(
      self,
      *,
      checkpoint_step: int | None = None,
      model_name: str | None = None,
      entry_filters: list[str] | None = None,
      entry_tags: list[str] | None = None,
      cameras: list[str] | None = None,
      model_save_dir: str | None = None,
      prediction_horizon: int | None = None,
      use_joint_torques: bool | None = None,
  ) -> rpc_api.StartExportResponse:
    """Start async model export from a specific checkpoint.

    Args:
      checkpoint_step: Export model from this checkpoint step. If None, uses
        the latest checkpoint.
      model_name: Optional name for the model to export.
      entry_filters: List of entry filters used to export the model. Required if
        model_name is provided. You do not need to provide ALL the prefixes
        used, you only need to provide one.
      entry_tags: Required data warehouse tags. Must match training.
      cameras: Camera names. Must match training.
      model_save_dir: Optional directory to where the model checkpoints are
        saved. If empty, uses default location.
      prediction_horizon: Optional prediction horizon to export the model with.
      use_joint_torques: Whether to include piper_joint_torques in proprio.

    Returns:
      Response containing error if export could not be started.
      Use get_export_status() to monitor progress.
    """
    query = rpc_api.StartExportQuery(
        checkpoint_step=checkpoint_step,
        model_name=model_name,
        entry_filters=entry_filters or [],
        entry_tags=entry_tags or [],
        cameras=cameras,
        model_save_dir=model_save_dir,
        prediction_horizon=prediction_horizon,
        use_joint_torques=use_joint_torques,
    )
    result = _rpc_call(self._rpc_client, "trainer.start_export", query)
    assert isinstance(result, rpc_api.StartExportResponse)
    return result

  def get_export_status(self) -> rpc_api.ExportStatusResponse:
    """Get the status of an ongoing or completed export.

    Returns:
      Response containing:
        - is_exporting: Whether export is in progress
        - is_finished: Whether export completed (success or failure)
        - error: Error message if export failed
        - model_id: The model ID if export completed successfully
    """
    result = _rpc_call(self._rpc_client, "trainer.get_export_status")
    assert isinstance(result, rpc_api.ExportStatusResponse)
    return result

  def list_checkpoints(self) -> rpc_api.ListCheckpointsResponse:
    """List available checkpoint steps.

    Returns:
      Response containing available checkpoint steps sorted ascending.
    """
    result = _rpc_call(self._rpc_client, "trainer.list_checkpoints")
    assert isinstance(result, rpc_api.ListCheckpointsResponse)
    return result


class ProgressPredictionTrainerClient:
  """Client for training progress prediction models.

  Trains models that predict task completion progress (0-1) from camera images.
  These models are used for behavior termination detection.

  The server automatically manages dataset caching based on entry filters:
  - First call with new filters: builds and caches the dataset
  - Subsequent calls: uses cached dataset if fresh, warns if stale
  - Use force_rebuild=True to get fresh data after a staleness warning

  Usage:
    # Start training from full episodes (single filter)
    response = robot.progress_trainer.train_model(
        model_name="pick_up_can_progress",
        entry_filters=["pick_up_can*"],
        training_steps=10000,
    )

    # Start training from multiple entry filters
    response = robot.progress_trainer.train_model(
        model_name="multi_task_progress",
        entry_filters=["pick_up_can*", "place_object*"],
        human_entry_filters=["dagger_*"],
        training_steps=10000,
    )

  Monitoring:
    # Poll for training completion
    while True:
        status = robot.progress_trainer.get_training_status()
        if status.is_finished:
            break
        print(f"Training: {status.steps_completed}/{status.max_steps}, "
              f"loss={status.loss:.4f}, acc={status.accuracy:.4f}")
        time.sleep(10.0)

    # To cancel training early:
    # robot.progress_trainer.cancel_training()
  """

  def __init__(self, rpc_client: client.BaseClient) -> None:
    """Initialize the client.

    Args:
      rpc_client: RPC client for server communication.
    """
    self._rpc_client = rpc_client

  def train_model(
      self,
      model_name: str,
      training_steps: int,
      entry_filters: list[str] | None = None,
      human_entry_filters: list[str] | None = None,
      force_rebuild: bool = False,
      batch_size: int = 32,
      task_type: str = "classification",
      cameras: list[str] | None = None,
      resume_from: str | None = None,
      checkpoint_interval_steps: int = 1000,
      max_checkpoints_to_keep: int = 10,
  ) -> rpc_api.StartProgressTrainingResponse:
    """Start training a progress prediction model.

    Initiates asynchronous model training on the server. Only one training
    run can be active at a time. Use get_training_status() to monitor
    progress and cancel_training() to stop early.

    At least one of entry_filters or human_entry_filters must be provided:
    - entry_filters: Processes full episodes matching the patterns
    - human_entry_filters: Extracts only human segments from matching episodes

    Args:
      model_name: Name for the exported model in the model warehouse.
      training_steps: Total number of training steps to run.
      entry_filters: Glob patterns for full episode entries
        (e.g., ["pick_up_can*", "place_object*"]).
      human_entry_filters: Glob patterns for human demonstration entries
        (e.g., ["dagger_*"]). Extracts only human segments.
      force_rebuild: If True, rebuild the dataset even if a fresh cache exists.
      batch_size: Training batch size.
      task_type: "classification" for binary done/not-done prediction,
        "regression" for continuous 0-1 progress.
      cameras: Camera names to use (e.g., ["wrist_camera"] or
        ["scene_camera", "wrist_camera"]). Required.
      resume_from: Checkpoint ID to resume from (e.g., "progress_model/20260202-150000").
        If None, starts fresh.
      checkpoint_interval_steps: Save checkpoint every N steps. Default 1000.
      max_checkpoints_to_keep: Max checkpoints to keep. Default 10.

    Returns:
      Response with dataset status and error field if training could not be
      started (e.g., another training run is already active).

    Raises:
      ValueError: If neither entry_filters nor human_entry_filters is provided.
      ValueError: If cameras is not provided.
      RuntimeError: If training is already running on the server.
    """
    if not entry_filters and not human_entry_filters:
      raise ValueError(
          "At least one of entry_filters or human_entry_filters must be provided"
      )

    if not cameras:
      raise ValueError(
          "cameras must be provided (e.g., ['wrist_camera'] or "
          "['scene_camera', 'wrist_camera'])"
      )

    # Check if training is already running
    if self.is_training_running():
      status = self.get_training_status()
      raise RuntimeError(
          f"Training is already running on the server "
          f"(step {status.steps_completed}/{status.max_steps}, loss={status.loss:.4f}). "
          f"Please either wait for it to finish or call cancel_training() first."
      )

    query = rpc_api.StartProgressTrainingQuery(
        model_name=model_name,
        training_steps=training_steps,
        entry_filters=entry_filters,
        human_entry_filters=human_entry_filters,
        force_rebuild=force_rebuild,
        batch_size=batch_size,
        task_type=task_type,
        cameras=cameras,
        resume_from=resume_from,
        checkpoint_interval_steps=checkpoint_interval_steps,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
    )
    result = _rpc_call(self._rpc_client, "trainer.train_progress_model", query)
    assert isinstance(result, rpc_api.StartProgressTrainingResponse)
    return result

  def is_training_running(self) -> bool:
    """Check if training is currently running on the server.

    Returns:
      True if training is in progress, False otherwise.
    """
    status = self.get_training_status()
    # Not running if finished OR failed
    if status.is_finished or status.phase in ("finished", "failed", "idle"):
      return False
    return True

  def get_training_status(self) -> rpc_api.ProgressTrainingStatusResponse:
    """Get information about the current progress prediction training status.

    Returns:
      Response containing:
        - is_finished: Whether training has completed
        - steps_completed: Current training step
        - max_steps: Total steps configured
        - loss: Current training loss
        - fps: Training speed (steps per second)
        - seconds_per_step: Time per training step
        - accuracy: Classification accuracy (if applicable)
        - f1: F1 score (if applicable)
    """
    result = _rpc_call(self._rpc_client, "trainer.get_progress_training_status")
    assert isinstance(result, rpc_api.ProgressTrainingStatusResponse)
    return result

  def cancel_training(self) -> rpc_api.CancelProgressTrainingResponse:
    """Cancel the current progress prediction training.

    Saves a checkpoint before stopping.

    Returns:
      Response indicating success or failure.
    """
    query = rpc_api.CancelProgressTrainingQuery()
    result = _rpc_call(
        self._rpc_client, "trainer.cancel_progress_training", query
    )
    assert isinstance(result, rpc_api.CancelProgressTrainingResponse)
    return result

  def reset_trainer(self) -> rpc_api.ResetTrainerResponse:
    """Reset the progress prediction trainer to initial state.

    Stops any running training and clears all state.

    Returns:
      Response indicating success or failure.
    """
    result = _rpc_call(self._rpc_client, "trainer.reset_progress_trainer")
    assert isinstance(result, rpc_api.ResetTrainerResponse)
    return result

  def start_export(
      self, checkpoint_step: int | None = None
  ) -> rpc_api.StartExportResponse:
    """Start async model export from a specific checkpoint.

    Args:
      checkpoint_step: Export model from this checkpoint step. If None, uses
        the latest checkpoint.

    Returns:
      Response containing error if export could not be started.
      Use get_export_status() to monitor progress.
    """
    query = rpc_api.StartExportQuery(checkpoint_step=checkpoint_step)
    result = _rpc_call(self._rpc_client, "trainer.start_progress_export", query)
    assert isinstance(result, rpc_api.StartExportResponse)
    return result

  def get_export_status(self) -> rpc_api.ExportStatusResponse:
    """Get the status of an ongoing or completed export.

    Returns:
      Response containing:
        - is_exporting: Whether export is in progress
        - is_finished: Whether export completed (success or failure)
        - error: Error message if export failed
        - model_id: The model ID if export completed successfully
    """
    result = _rpc_call(self._rpc_client, "trainer.get_progress_export_status")
    assert isinstance(result, rpc_api.ExportStatusResponse)
    return result

  def list_checkpoints(self) -> rpc_api.ListCheckpointsResponse:
    """List available checkpoint steps.

    Returns:
      Response containing available checkpoint steps sorted ascending.
    """
    result = _rpc_call(self._rpc_client, "trainer.list_progress_checkpoints")
    assert isinstance(result, rpc_api.ListCheckpointsResponse)
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

  def visual_trajectory_motion(
      self,
      visual_trajectory_name: str,
      timeout: float | None = None,
      static_gripper: bool = False,
      motion_type: rpc_api.TrajectoryMotionType = rpc_api.TrajectoryMotionType.FULL,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Execute a visual trajectory motion and return a future.

    Args:
      visual_trajectory_name: Name of the visual trajectory in the library.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      static_gripper: Whether to keep the gripper static.
      motion_type: FULL plays the entire trajectory. GO_TO_START uses visual
        servoing to move to the first frame. GO_TO_END is not supported.
    """
    return self._behaviour_client.visual_trajectory_motion(
        visual_trajectory_name=visual_trajectory_name,
        timeout=timeout,
        arm=self._arm,
        static_gripper=static_gripper,
        motion_type=motion_type,
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
      training_server_address: str,
      use_compression: bool = False,
      timeout: int = 5000,
      query_timeout: int | None = None,
      primary_arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ):
    """Initialize the robot client.

    All sub-clients are created lazily on first access, except for the main
    server client which is created eagerly to verify connectivity.

    Args:
      server_address: Main RPC server address (e.g., "tcp://host:7532").
      query_server_address: Query server address (e.g., "tcp://host:7533").
      training_server_address: Training server address (e.g., "tcp://host:7534").
        This server handles both imitation learning and progress prediction.
      use_compression: Whether to compress RPC payloads with zstd.
      timeout: Default timeout in milliseconds for RPC calls.
      query_timeout: Timeout for query server, defaults to timeout if None.
      primary_arm: Arm selection exposed by the `arm` property.
    """
    # Store configuration for lazy client creation
    self._server_address = server_address
    self._query_server_address = query_server_address
    self._training_server_address = training_server_address
    self._use_compression = use_compression
    self._timeout = timeout
    self._query_timeout = (
        query_timeout if query_timeout is not None else timeout
    )
    self._primary_arm_side = primary_arm

    # Eagerly create and ping the main server client to verify connectivity
    self._base_client = client.BaseClient(
        server_address,
        use_compression=use_compression,
        timeout=timeout,
        service_name="rpc server",
    )

  @functools.cached_property
  def _query_client(self) -> client.BaseClient:
    """Lazily create the query server client."""
    return client.BaseClient(
        self._query_server_address,
        use_compression=self._use_compression,
        timeout=self._query_timeout,
        service_name="query server",
    )

  @functools.cached_property
  def _training_client(self) -> client.BaseClient:
    """Lazily create the training server client."""
    return client.BaseClient(
        self._training_server_address,
        use_compression=self._use_compression,
        timeout=self._timeout,
        service_name="training server",
    )

  def _make_behaviour_client(self) -> client.BaseClient:
    """Create a new behaviour client (used for concurrent operations)."""
    return client.BaseClient(
        self._server_address,
        use_compression=self._use_compression,
        timeout=self._timeout,
        service_name="main server",
    )

  # --- Public sub-clients (lazy via cached_property) ---

  @functools.cached_property
  def exec_mode(self) -> ExecModeClient:
    """Client for execution mode control."""
    return ExecModeClient(self._base_client)

  @functools.cached_property
  def raw_robot(self) -> RawRobotClient:
    """Client for raw sensor data access."""
    return RawRobotClient(self._base_client)

  @functools.cached_property
  def query(self) -> QueryClient:
    """Client for synchronous query operations."""
    return QueryClient(self._query_client)

  @functools.cached_property
  def recording(self) -> RecordingClient:
    """Client for trajectory recording."""
    return RecordingClient(self._base_client)

  @functools.cached_property
  def visual_trajectory_recording(self) -> VisualRecordingClient:
    """Client for visual trajectory recording."""
    return VisualRecordingClient(self._base_client)

  @functools.cached_property
  def episode_observer(self) -> EpisodeObserverClient:
    """Client for episode recording observer (data gathering UI)."""
    return EpisodeObserverClient(self._base_client)

  @functools.cached_property
  def collect_data(self) -> CollectDataClient:
    """Client for collect-data workflow orchestration."""
    return CollectDataClient(self._base_client)

  @functools.cached_property
  def dagger(self) -> DaggerClient:
    """Client for DAgger policy-assist orchestration."""
    return DaggerClient(self._base_client)

  @functools.cached_property
  def eval(self) -> EvalClient:
    """Client for blinded model evaluation orchestration."""
    return EvalClient(self._base_client)

  @functools.cached_property
  def hardware_health(self) -> HardwareHealthClient:
    """Client for hardware health status."""
    return HardwareHealthClient(self._base_client)

  @functools.cached_property
  def object_library(self) -> ObjectLibraryClient:
    """Client for object library management."""
    return ObjectLibraryClient(self._base_client)

  @functools.cached_property
  def trajectory_library(self) -> TrajectoryLibraryClient:
    """Client for trajectory library management."""
    return TrajectoryLibraryClient(self._base_client)

  @functools.cached_property
  def visual_pose_library(self) -> VisualPoseLibraryClient:
    """Client for visual pose library management."""
    return VisualPoseLibraryClient(self._base_client)

  @functools.cached_property
  def visual_trajectory_library(self) -> VisualTrajectoryLibraryClient:
    """Client for visual trajectory library management."""
    return VisualTrajectoryLibraryClient(self._base_client)

  @functools.cached_property
  def apriltag(self) -> AprilTagClient:
    """Client for AprilTag detection services."""
    return AprilTagClient(self._base_client)

  @functools.cached_property
  def model_services(self) -> ModelServicesClient:
    """Client for managing model inference services."""
    return ModelServicesClient(self._training_client)

  @functools.cached_property
  def behaviour(self) -> BehaviourClient:
    """Client for behaviour execution."""
    return BehaviourClient(self._make_behaviour_client)

  @functools.cached_property
  def trainer(self) -> TrainerClient:
    """Client for model training."""
    return TrainerClient(self._training_client)

  @functools.cached_property
  def progress_trainer(self) -> ProgressPredictionTrainerClient:
    """Client for progress prediction model training."""
    return ProgressPredictionTrainerClient(self._training_client)

  @functools.cached_property
  def left_arm(self) -> ArmClient:
    """Client for left arm control."""
    return ArmClient(
        self.behaviour,
        sdk_futures.ArmSide.LEFT,
        query_client=self.query,
    )

  @functools.cached_property
  def right_arm(self) -> ArmClient:
    """Client for right arm control."""
    return ArmClient(
        self.behaviour,
        sdk_futures.ArmSide.RIGHT,
        query_client=self.query,
    )

  @property
  def arm(self) -> ArmClient:
    """Primary arm client for single-arm usage."""
    if self._primary_arm_side == sdk_futures.ArmSide.LEFT:
      return self.left_arm
    elif self._primary_arm_side == sdk_futures.ArmSide.RIGHT:
      return self.right_arm
    else:
      raise ValueError(f"unsupported primary arm: {self._primary_arm_side}")

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
      camera_response = self.raw_robot.get_camera_data(camera=camera)
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
        camera_type=camera,
    )
    return self.visual_pose_library.add_entry(
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
    camera_data = self.raw_robot.get_camera_data(camera=camera)
    if camera_data.rgb is None:
      detections = rpc_api.AprilTagDetectResponse(
          detections=[],
          error="camera frame unavailable",
      )
      return AprilTagCameraDetection(
          camera_data=camera_data,
          detections=detections,
      )

    detections = self.apriltag.detect(
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
    response = self.exec_mode.set_execution_mode(
        new_mode=rpc_api.ExecutionMode.READY
    )
    confirmed = self.exec_mode.get_execution_mode()
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
    response = self.exec_mode.set_execution_mode(
        new_mode=rpc_api.ExecutionMode.STOP
    )
    confirmed = self.exec_mode.get_execution_mode()
    if confirmed.current_mode != rpc_api.ExecutionMode.STOP:
      raise RuntimeError(
          f"failed to set execution mode to STOP (got {confirmed.current_mode})"
      )
    return response
