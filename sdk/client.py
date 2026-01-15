"""High-level clients for robot control and behaviour execution."""

import pickle
import threading
import time
from typing import Any, Callable, Sequence

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
      self, query: rpc_api.ExecutionModeQuery
  ) -> rpc_api.ExecutionModeQueryResponse:
    """Set the execution mode.

    Args:
      query: Query containing the target execution mode.

    Returns:
      Response containing the new current mode.
    """
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
      self, camera: rpc_api.CameraQuery
  ) -> rpc_api.CameraQueryResponse:
    """Get RGB and depth data from a camera.

    Args:
      camera: Query specifying which camera to read from.

    Returns:
      Response containing RGB, depth, and intrinsics data.
    """
    result = _rpc_call(self._rpc_client, "raw_robot.get_camera_data", camera)
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
    robot.trajectories.add(rpc_api.AddTrajectoryQuery(trajectory=trajectory))
  """

  def __init__(self, rpc_client: client.BaseClient):
    self._rpc_client = rpc_client

  def prepare(
      self,
      query: rpc_api.PrepareRecordingQuery | None = None,
  ) -> rpc_api.PrepareRecordingResponse:
    """Prepare for recording with specified trajectory type and execution mode.

    This clears any previously recorded trajectory. The robot will be switched
    to the specified execution mode (TEACH or TELEOP) if not already.

    Args:
      query: Recording configuration. If None, uses defaults (JOINT_ABSOLUTE
        trajectory type, TEACH mode, 30s timeout).

    Returns:
      Response with error field set if preparation failed.
    """
    if query is None:
      query = rpc_api.PrepareRecordingQuery()
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

  def __init__(self, rpc_client: client.BaseClient):
    self._rpc_client = rpc_client

  def start(self) -> None:
    """Start episode recording."""
    _rpc_call(self._rpc_client, "episode_observer.start")

  def stop(self) -> None:
    """Stop episode recording."""
    _rpc_call(self._rpc_client, "episode_observer.stop")

  def save(self) -> None:
    """Save the current episode."""
    _rpc_call(self._rpc_client, "episode_observer.save")

  def discard(self) -> None:
    """Discard the current episode."""
    _rpc_call(self._rpc_client, "episode_observer.discard")

  def get_state(self) -> rpc_api.EpisodeObserverStateResponse:
    """Get the current episode observer state."""
    result = _rpc_call(self._rpc_client, "episode_observer.get_state")
    assert isinstance(result, rpc_api.EpisodeObserverStateResponse)
    return result

  def set_task_description(
      self, query: rpc_api.SetTaskDescriptionQuery
  ) -> None:
    """Set the task description for the current episode.

    Args:
      query: Query containing the task description text.
    """
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

  def delete_entry(
      self, entry: rpc_api.DeleteObjectQuery
  ) -> rpc_api.DeleteObjectQueryResponse:
    """Delete an object from the library.

    Args:
      entry: Query containing the object name to delete.
    """
    result = _rpc_call(self._rpc_client, "object_library.delete_entry", entry)
    assert isinstance(result, rpc_api.DeleteObjectQueryResponse)
    return result

  def segment_object(
      self,
      query: rpc_api.ObjectSegmentationQuery,
      timeout: int | None = None,
  ) -> rpc_api.ObjectSegmentationQueryResponse:
    """Segment an object from video frames using point prompts.

    Args:
      query: Segmentation query with frames and point prompts.
      timeout: RPC timeout in milliseconds, or None for default.
    """
    result = _rpc_call(
        self._rpc_client, "object_library.segment_object", query, timeout
    )
    assert isinstance(result, rpc_api.ObjectSegmentationQueryResponse)
    return result

  def add_object_views(
      self,
      query: rpc_api.AddObjectViewsQuery,
      timeout: int | None = None,
  ) -> rpc_api.AddObjectViewsQueryResponse:
    """Add views of an object to the library for recognition training.

    Args:
      query: Query with object name, frames, and segmentation masks.
      timeout: RPC timeout in milliseconds, or None for default.
    """
    result = _rpc_call(
        self._rpc_client, "object_library.add_object_views", query, timeout
    )
    assert isinstance(result, rpc_api.AddObjectViewsQueryResponse)
    return result

  def get_heatmap(
      self,
      query: rpc_api.ObjectHeatmapQuery,
      timeout: int | None = None,
  ) -> rpc_api.ObjectHeatmapResponse:
    """Get a live detection heatmap for an object.

    Args:
      query: Query containing the object name.
      timeout: RPC timeout in milliseconds, or None for default.
    """
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
      self, entry: rpc_api.AddTrajectoryQuery
  ) -> rpc_api.AddTrajectoryQueryResponse:
    """Add a trajectory to the library.

    Args:
      entry: Query containing the trajectory to add.
    """
    result = _rpc_call(self._rpc_client, "trajectory_library.add_entry", entry)
    assert isinstance(result, rpc_api.AddTrajectoryQueryResponse)
    return result

  def delete_entry(
      self, entry: rpc_api.DeleteTrajectoryQuery
  ) -> rpc_api.DeleteTrajectoryQueryResponse:
    """Delete a trajectory from the library.

    Args:
      entry: Query containing the trajectory name to delete.
    """
    result = _rpc_call(
        self._rpc_client, "trajectory_library.delete_entry", entry
    )
    assert isinstance(result, rpc_api.DeleteTrajectoryQueryResponse)
    return result

  def load_entry(
      self, query: rpc_api.LoadTrajectoryQuery
  ) -> rpc_api.LoadTrajectoryQueryResponse:
    """Load a trajectory from the library by name.

    Args:
      query: Query containing the trajectory name to load.
    """
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
      self, entry: rpc_api.AddVisualPoseQuery
  ) -> rpc_api.AddVisualPoseQueryResponse:
    """Add a visual pose to the library.

    Args:
      entry: Query containing the visual pose to add.
    """
    result = _rpc_call(self._rpc_client, "visual_pose_library.add_entry", entry)
    assert isinstance(result, rpc_api.AddVisualPoseQueryResponse)
    return result

  def delete_entry(
      self, entry: rpc_api.DeleteVisualPoseQuery
  ) -> rpc_api.DeleteVisualPoseQueryResponse:
    """Delete a visual pose from the library.

    Args:
      entry: Query containing the pose name to delete.
    """
    result = _rpc_call(
        self._rpc_client, "visual_pose_library.delete_entry", entry
    )
    assert isinstance(result, rpc_api.DeleteVisualPoseQueryResponse)
    return result

  def load_entry(
      self, query: rpc_api.LoadVisualPoseQuery
  ) -> rpc_api.LoadVisualPoseQueryResponse:
    """Load a visual pose from the library by name.

    Args:
      query: Query containing the pose name to load.
    """
    result = _rpc_call(
        self._rpc_client, "visual_pose_library.load_entry", query
    )
    assert isinstance(result, rpc_api.LoadVisualPoseQueryResponse)
    return result

  def segment_reference(
      self,
      query: rpc_api.VisualReferenceSegmentationQuery,
      timeout: int | None = None,
  ) -> rpc_api.VisualReferenceSegmentationQueryResponse:
    """Segment a visual reference from a single frame using point prompts.

    Args:
      query: Segmentation query with frame and point prompts.
      timeout: RPC timeout in milliseconds, or None for default.
    """
    result = _rpc_call(
        self._rpc_client,
        "visual_pose_library.segment_reference",
        query,
        timeout,
    )
    assert isinstance(result, rpc_api.VisualReferenceSegmentationQueryResponse)
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
                on_cancel=lambda: self.cancel_behaviour(
                    rpc_api.CancelTicketQuery(ticket_id=response.ticket_id)
                )
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
            on_cancel=lambda: self.cancel_behaviour(
                rpc_api.CancelTicketQuery(ticket_id=response.ticket_id)
            )
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
          self.cancel_behaviour(
              rpc_api.CancelTicketQuery(ticket_id=ticket_holder["ticket_id"])
          )
        except Exception:  # pylint: disable=broad-except
          pass

    return self._executor.submit_for_arm(
        arm, _task, cancel_callback=_cancel_callback
    )

  def cancel_behaviour(
      self, query: rpc_api.CancelTicketQuery
  ) -> rpc_api.CancelTicketResponse:
    """Cancel a running behaviour by ticket ID.

    Args:
      query: Query containing the ticket ID to cancel.
    """
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
      query: rpc_api.TrajectoryMotionQuery,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate trajectory motion. Returns immediately with ticket_id.

    Args:
      query: Query specifying trajectory name and motion parameters.
    """
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.trajectory_motion", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_open_gripper(
      self,
      query: rpc_api.OpenGripperQuery | None = None,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate open gripper. Returns immediately with ticket_id.

    Args:
      query: Query with target position, or None for default (fully open).
    """
    query = query or rpc_api.OpenGripperQuery()
    result = _rpc_call(self._get_rpc_client(), "behaviour.open_gripper", query)
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_close_gripper(
      self,
      query: rpc_api.CloseGripperQuery | None = None,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate close gripper. Returns immediately with ticket_id.

    Args:
      query: Query with target position, or None for default (fully closed).
    """
    query = query or rpc_api.CloseGripperQuery()
    result = _rpc_call(self._get_rpc_client(), "behaviour.close_gripper", query)
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_go_to_joints(
      self,
      query: rpc_api.GoToJointsQuery,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate go to joints. Returns immediately with ticket_id.

    Args:
      query: Query with target joint configuration.
    """
    result = _rpc_call(self._get_rpc_client(), "behaviour.go_to_joints", query)
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_go_to_neutral_pose(
      self,
      query: rpc_api.GoToNeutralPoseQuery | None = None,
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate go to neutral pose. Returns immediately with ticket_id.

    Args:
      query: Query options, or None for defaults.
    """
    query = query or rpc_api.GoToNeutralPoseQuery()
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.go_to_neutral_pose", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  def initiate_wait_for_object(
      self, query: rpc_api.WaitForObjectQuery
  ) -> rpc_api.BehaviourInitiatedResponse:
    """Initiate wait for object. Returns immediately with ticket_id.

    Args:
      query: Query with object names and timeout.
    """
    result = _rpc_call(
        self._get_rpc_client(), "behaviour.wait_for_object", query
    )
    assert isinstance(result, rpc_api.BehaviourInitiatedResponse)
    return result

  # non-blocking convenience methods that return futures

  def trajectory_motion(
      self,
      query: rpc_api.TrajectoryMotionQuery,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue trajectory motion and return a future.

    Args:
      query: Query specifying trajectory name and motion parameters.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_trajectory_motion(query),
        timeout=timeout,
        arm=arm,
        behaviour_type="trajectory_motion",
    )

  def open_gripper(
      self,
      query: rpc_api.OpenGripperQuery | None = None,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue open gripper and return a future.

    Args:
      query: Query with target position, or None for default (fully open).
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    final_query = query or rpc_api.OpenGripperQuery()
    return self._submit_behaviour(
        lambda: self.initiate_open_gripper(final_query),
        timeout=timeout,
        arm=arm,
        behaviour_type="open_gripper",
    )

  def close_gripper(
      self,
      query: rpc_api.CloseGripperQuery | None = None,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue close gripper and return a future.

    Args:
      query: Query with target position, or None for default (fully closed).
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    final_query = query or rpc_api.CloseGripperQuery()
    return self._submit_behaviour(
        lambda: self.initiate_close_gripper(final_query),
        timeout=timeout,
        arm=arm,
        behaviour_type="close_gripper",
    )

  def go_to_joints(
      self,
      query: rpc_api.GoToJointsQuery,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue go to joints and return a future.

    Args:
      query: Query with target joint configuration.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    return self._submit_behaviour(
        lambda: self.initiate_go_to_joints(query),
        timeout=timeout,
        arm=arm,
    )

  def go_to_neutral_pose(
      self,
      query: rpc_api.GoToNeutralPoseQuery | None = None,
      timeout: float | None = None,
      arm: sdk_futures.ArmSide = sdk_futures.ArmSide.LEFT,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Enqueue go to neutral pose and return a future.

    Args:
      query: Query options, or None for defaults.
      timeout: Maximum seconds to wait for completion, or None for no limit.
      arm: Which arm this behaviour requires.
    """
    final_query = query or rpc_api.GoToNeutralPoseQuery()
    return self._submit_behaviour(
        lambda: self.initiate_go_to_neutral_pose(final_query),
        timeout=timeout,
        arm=arm,
        behaviour_type="go_to_neutral_pose",
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
    query = rpc_api.WaitForObjectQuery(
        object_names=list(object_names), timeout_seconds=timeout_seconds
    )
    return self._submit_behaviour(
        lambda: self.initiate_wait_for_object(query),
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
      query: rpc_api.TrajectoryMotionQuery,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Execute a trajectory motion and return a future.

    Args:
      query: Query specifying trajectory name and motion parameters.
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.trajectory_motion(
        query=query, timeout=timeout, arm=self._arm
    )

  def open_gripper(
      self,
      query: rpc_api.OpenGripperQuery | None = None,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Open the gripper and return a future.

    Args:
      query: Query with target position, or None for default (fully open).
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.open_gripper(
        query=query, timeout=timeout, arm=self._arm
    )

  def close_gripper(
      self,
      query: rpc_api.CloseGripperQuery | None = None,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Close the gripper and return a future.

    Args:
      query: Query with target position, or None for default (fully closed).
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.close_gripper(
        query=query, timeout=timeout, arm=self._arm
    )

  def go_to_neutral_pose(
      self,
      query: rpc_api.GoToNeutralPoseQuery | None = None,
      timeout: float | None = None,
  ) -> sdk_futures.Future[rpc_api.TicketStatusResponse]:
    """Move the arm to a neutral pose and return a future.

    Args:
      query: Query options, or None for defaults.
      timeout: Maximum seconds to wait for completion, or None for no limit.
    """
    return self._behaviour_client.go_to_neutral_pose(
        query=query, timeout=timeout, arm=self._arm
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
    robot.left_arm.open_gripper().result()
  """

  def __init__(
      self,
      server_address: str,
      query_server_address: str,
      use_compression: bool = False,
      timeout: int = 5000,
      query_timeout: int | None = None,
  ):
    """Initialize the robot client.

    Args:
      server_address: Main RPC server address (e.g., "tcp://host:7532").
      query_server_address: Query server address (e.g., "tcp://host:7533").
      use_compression: Whether to compress RPC payloads with zstd.
      timeout: Default timeout in milliseconds for RPC calls.
      query_timeout: Timeout for query server, defaults to timeout if None.
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
  def behaviour(self) -> BehaviourClient:
    """Client for behaviour execution."""
    return self._behaviour

  @property
  def left_arm(self) -> "ArmClient":
    """Arm-scoped client for left arm operations."""
    return self._left_arm

  @property
  def right_arm(self) -> "ArmClient":
    """Arm-scoped client for right arm operations."""
    return self._right_arm

  def activate(self) -> rpc_api.ExecutionModeQueryResponse:
    """Set the robot to READY mode for accepting behaviour commands.

    Raises:
      RuntimeError: If the mode transition fails.
    """
    response = self._exec_mode.set_execution_mode(
        query=rpc_api.ExecutionModeQuery(new_mode=rpc_api.ExecutionMode.READY)
    )
    confirmed = self._exec_mode.get_execution_mode()
    if confirmed.current_mode != rpc_api.ExecutionMode.READY:
      raise RuntimeError(
          f"failed to set execution mode to READY (got {confirmed.current_mode})"
      )
    return response

  def deactivate(self) -> rpc_api.ExecutionModeQueryResponse:
    """Set the robot to STOP mode, parking the arm at zero position.

    Raises:
      RuntimeError: If the mode transition fails.
    """
    response = self._exec_mode.set_execution_mode(
        query=rpc_api.ExecutionModeQuery(new_mode=rpc_api.ExecutionMode.STOP)
    )
    confirmed = self._exec_mode.get_execution_mode()
    if confirmed.current_mode != rpc_api.ExecutionMode.STOP:
      raise RuntimeError(
          f"failed to set execution mode to STOP (got {confirmed.current_mode})"
      )
    return response
