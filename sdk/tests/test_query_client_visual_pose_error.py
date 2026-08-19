"""Tests for QueryClient.get_visual_pose_error."""

import pickle

import numpy as np
import pytest

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


class _RecordingRpcClient:
  """Fake RPC client capturing the call and returning a pickled response."""

  def __init__(self, response: object) -> None:
    self._response = pickle.dumps(response)
    self.fn_name: str | None = None
    self.data: bytes | None = None
    self.timeout: int | None = None

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    self.fn_name = fn_name
    self.data = data
    self.timeout = timeout
    return self._response


def _success_response() -> rpc_api.VisualPoseErrorResponse:
  return rpc_api.VisualPoseErrorResponse(
      pose_base_camcur_position=np.array([0.1, 0.2, 0.3]),
      pose_base_camcur_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
      pose_camcur_camtarget_position=np.array([0.01, -0.02, 0.03]),
      pose_camcur_camtarget_quaternion=np.array([0.0, 0.0, 1.0, 0.0]),
      reference_type=rpc_api.VisualReference.OBJECT,
      num_inliers=42,
      pcl_converged=True,
      pcl_mean_error=0.005,
      frame_timestamp=123.456,
  )


def test_get_visual_pose_error_fn_name_and_query() -> None:
  rpc_client = _RecordingRpcClient(_success_response())
  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  query_client.get_visual_pose_error("dock_pose")
  assert rpc_client.fn_name == "query.visual_pose_error"
  assert rpc_client.data is not None
  query = pickle.loads(rpc_client.data)
  assert isinstance(query, rpc_api.VisualPoseErrorQuery)
  assert query.visual_pose_name == "dock_pose"


def test_get_visual_pose_error_forwards_timeout_with_buffer() -> None:
  rpc_client = _RecordingRpcClient(_success_response())
  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  query_client.get_visual_pose_error("dock_pose", timeout_seconds=5.0)
  expected = int(sdk_client._with_buffer(5.0) * 1000)  # pylint: disable=protected-access
  assert rpc_client.timeout == expected


def test_get_visual_pose_error_default_timeout() -> None:
  rpc_client = _RecordingRpcClient(_success_response())
  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  query_client.get_visual_pose_error("dock_pose")
  expected = int(sdk_client._with_buffer(30.0) * 1000)  # pylint: disable=protected-access
  assert rpc_client.timeout == expected


def test_get_visual_pose_error_returns_response() -> None:
  rpc_client = _RecordingRpcClient(_success_response())
  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  result = query_client.get_visual_pose_error("dock_pose")
  assert isinstance(result, rpc_api.VisualPoseErrorResponse)
  assert result.pose_base_camcur_position is not None
  assert result.pose_camcur_camtarget_quaternion is not None
  np.testing.assert_array_equal(
      result.pose_base_camcur_position, np.array([0.1, 0.2, 0.3])
  )
  np.testing.assert_array_equal(
      result.pose_camcur_camtarget_quaternion, np.array([0.0, 0.0, 1.0, 0.0])
  )
  assert result.reference_type == rpc_api.VisualReference.OBJECT
  assert result.num_inliers == 42
  assert result.pcl_converged is True
  assert result.pcl_mean_error == 0.005
  assert result.frame_timestamp == 123.456
  assert result.error is None


def test_get_visual_pose_error_server_error_returned_not_raised() -> None:
  response = rpc_api.VisualPoseErrorResponse(error="unknown pose: dock_pose")
  rpc_client = _RecordingRpcClient(response)
  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  result = query_client.get_visual_pose_error("dock_pose")
  assert result.error == "unknown pose: dock_pose"
  assert result.pose_base_camcur_position is None
  assert result.pose_camcur_camtarget_position is None


def test_get_visual_pose_error_wrong_response_type_asserts() -> None:
  rpc_client = _RecordingRpcClient(rpc_api.CanSeeObjectResponse(visible=True))
  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  with pytest.raises(AssertionError):
    query_client.get_visual_pose_error("dock_pose")
