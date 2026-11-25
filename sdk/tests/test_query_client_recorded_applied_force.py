"""Tests for QueryClient.get_recorded_applied_force."""

import pickle

import numpy as np

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


def test_get_recorded_applied_force_asks_the_robot_for_one_run() -> None:
  answer = rpc_api.RecordedAppliedForceResponse(
      visual_trajectory_name="Nero_open_cover",
      motion_type=rpc_api.TrajectoryMotionType.FULL,
      applied_forces=np.array([0.0, 12.5, 33.6]),
      step_seconds=0.1,
  )
  calls: list[tuple[str, bytes | None]] = []

  def rpc_client(
      fn_name: str, data: bytes | None = None, timeout: int | None = None
  ) -> bytes:
    del timeout
    calls.append((fn_name, data))
    return pickle.dumps(answer)

  query_client = sdk_client.QueryClient(rpc_client)  # type: ignore[arg-type]
  response = query_client.get_recorded_applied_force("34e02d95-e1f3")

  ((fn_name, data),) = calls
  assert fn_name == "query.recorded_applied_force"
  assert data is not None
  assert pickle.loads(data) == rpc_api.RecordedAppliedForceQuery(
      entry_id="34e02d95-e1f3"
  )
  assert response.applied_forces is not None
  assert response.applied_forces.max() == 33.6
