"""Wire-shape tests for the visual pose error query/response dataclasses."""

import pickle

import numpy as np

from r2_labs.sdk import rpc_api


def test_query_pickle_round_trip():
  query = rpc_api.VisualPoseErrorQuery(visual_pose_name="shelf_left")
  restored = pickle.loads(pickle.dumps(query))
  assert restored.visual_pose_name == "shelf_left"


def test_response_pickle_round_trip_preserves_numpy_and_enum_fields():
  response = rpc_api.VisualPoseErrorResponse(
      pose_base_camcur_position=np.array([0.1, -0.2, 0.3], dtype=np.float64),
      pose_base_camcur_quaternion=np.array(
          [1.0, 0.0, 0.0, 0.0], dtype=np.float64
      ),
      pose_camcur_camtarget_position=np.array(
          [0.01, 0.02, 0.03], dtype=np.float64
      ),
      pose_camcur_camtarget_quaternion=np.array(
          [0.0, 0.0, 1.0, 0.0], dtype=np.float64
      ),
      reference_type=rpc_api.VisualReference.OBJECT,
      num_inliers=42,
      pcl_converged=True,
      pcl_mean_error=0.005,
      frame_timestamp=1234.5,
  )
  restored = pickle.loads(pickle.dumps(response))

  # the pose fields are declared optional; the round trip must keep them set
  assert response.pose_base_camcur_position is not None
  assert response.pose_base_camcur_quaternion is not None
  assert response.pose_camcur_camtarget_position is not None
  assert response.pose_camcur_camtarget_quaternion is not None

  np.testing.assert_array_equal(
      restored.pose_base_camcur_position, response.pose_base_camcur_position
  )
  np.testing.assert_array_equal(
      restored.pose_base_camcur_quaternion,
      response.pose_base_camcur_quaternion,
  )
  np.testing.assert_array_equal(
      restored.pose_camcur_camtarget_position,
      response.pose_camcur_camtarget_position,
  )
  np.testing.assert_array_equal(
      restored.pose_camcur_camtarget_quaternion,
      response.pose_camcur_camtarget_quaternion,
  )
  assert restored.pose_base_camcur_position.dtype == np.float64
  assert restored.pose_base_camcur_quaternion.dtype == np.float64
  assert restored.pose_camcur_camtarget_position.dtype == np.float64
  assert restored.pose_camcur_camtarget_quaternion.dtype == np.float64
  assert restored.reference_type is rpc_api.VisualReference.OBJECT
  assert restored.num_inliers == 42
  assert restored.pcl_converged is True
  assert restored.pcl_mean_error == 0.005
  assert restored.frame_timestamp == 1234.5
  assert restored.error is None


def test_response_all_none_error_shape_round_trips():
  # Endpoint error paths ship a response with only `error` set; every other
  # field must arrive as None.
  response = rpc_api.VisualPoseErrorResponse(error="unknown pose: shelf_left")
  restored = pickle.loads(pickle.dumps(response))

  assert restored.error == "unknown pose: shelf_left"
  assert restored.pose_base_camcur_position is None
  assert restored.pose_base_camcur_quaternion is None
  assert restored.pose_camcur_camtarget_position is None
  assert restored.pose_camcur_camtarget_quaternion is None
  assert restored.reference_type is None
  assert restored.num_inliers is None
  assert restored.pcl_converged is None
  assert restored.pcl_mean_error is None
  assert restored.frame_timestamp is None


def test_response_minimal_construction_defaults_to_none():
  response = rpc_api.VisualPoseErrorResponse(error="camera unavailable")

  assert response.error == "camera unavailable"
  assert response.pose_base_camcur_position is None
  assert response.pose_base_camcur_quaternion is None
  assert response.pose_camcur_camtarget_position is None
  assert response.pose_camcur_camtarget_quaternion is None
  assert response.reference_type is None
  assert response.num_inliers is None
  assert response.pcl_converged is None
  assert response.pcl_mean_error is None
  assert response.frame_timestamp is None


def test_apriltag_response_carries_none_quality_fields():
  # AprilTag measurements have no UFM/PCL quality metrics; the response must
  # represent that as None alongside real pose data.
  response = rpc_api.VisualPoseErrorResponse(
      pose_base_camcur_position=np.zeros(3),
      pose_base_camcur_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
      pose_camcur_camtarget_position=np.zeros(3),
      pose_camcur_camtarget_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
      reference_type=rpc_api.VisualReference.APRILTAG,
      frame_timestamp=99.0,
  )
  restored = pickle.loads(pickle.dumps(response))

  assert restored.reference_type is rpc_api.VisualReference.APRILTAG
  assert restored.num_inliers is None
  assert restored.pcl_converged is None
  assert restored.pcl_mean_error is None
  assert restored.error is None
