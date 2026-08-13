"""Tests for how the SDK expresses a cartesian tooltip pose on the wire."""

import pickle

import numpy as np
import pytest

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api

_INITIATE_FN = "behaviour.go_to_pose"
_STATUS_FN = "behaviour.ticket_status"
_POSITION = [0.1, -0.4, 1.0]
_ROTATION = [0.70710678, 0.70710678, 0.0, 0.0]


class _CapturingRpcClient:
  """Fake RPC client that records the query of every call it serves."""

  def __init__(self) -> None:
    self.queries: dict[str, object] = {}

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    del timeout
    self.queries[fn_name] = pickle.loads(data) if data is not None else None
    if fn_name == _STATUS_FN:
      return pickle.dumps(
          rpc_api.TicketStatusResponse(
              info=rpc_api.TicketInfo(
                  ticket_id="t1",
                  status=rpc_api.TicketStatus.COMPLETED,
                  behaviour_type="go_to_pose",
                  created_at=0.0,
              )
          )
      )
    return pickle.dumps(rpc_api.BehaviourInitiatedResponse(ticket_id="t1"))


def _make_client() -> tuple[sdk_client.BehaviourClient, _CapturingRpcClient]:
  fake = _CapturingRpcClient()
  client = sdk_client.BehaviourClient(lambda: fake)  # type: ignore[arg-type]
  return client, fake


def _sent_query(fake: _CapturingRpcClient) -> rpc_api.GoToPoseQuery:
  query = fake.queries[_INITIATE_FN]
  assert isinstance(query, rpc_api.GoToPoseQuery)
  return query


def test_pose_is_sent_as_position_and_quaternion() -> None:
  client, fake = _make_client()
  client.initiate_go_to_pose(position=_POSITION, rotation=_ROTATION)
  query = _sent_query(fake)
  np.testing.assert_allclose(query.position, _POSITION)
  np.testing.assert_allclose(query.rotation, _ROTATION)
  assert query.rotation_tolerance is None


def test_rotation_tolerance_is_sent_through() -> None:
  client, fake = _make_client()
  client.initiate_go_to_pose(
      position=_POSITION, rotation=_ROTATION, rotation_tolerance=0.25
  )
  assert _sent_query(fake).rotation_tolerance == 0.25


def test_rotation_is_required() -> None:
  client, _ = _make_client()
  with pytest.raises(TypeError):
    client.go_to_pose(position=_POSITION)  # type: ignore[call-arg]


def test_the_future_carries_the_pose_to_the_server() -> None:
  client, fake = _make_client()
  result = client.go_to_pose(position=_POSITION, rotation=_ROTATION).result()
  assert result.info is not None
  assert result.info.status is rpc_api.TicketStatus.COMPLETED
  query = _sent_query(fake)
  np.testing.assert_allclose(query.position, _POSITION)
  np.testing.assert_allclose(query.rotation, _ROTATION)
