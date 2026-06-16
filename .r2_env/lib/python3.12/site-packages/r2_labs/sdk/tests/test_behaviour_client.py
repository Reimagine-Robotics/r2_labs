"""Tests for BehaviourClient failure propagation through futures."""

import pickle

import pytest

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api

_INITIATE_FN = "behaviour.trajectory_motion"
_STATUS_FN = "behaviour.ticket_status"


class _DispatchingRpcClient:
  """Fake RPC client returning a different pickled response per fn_name."""

  def __init__(self, responses: dict[str, object]) -> None:
    self._responses: dict[str, bytes] = {
        fn_name: pickle.dumps(resp) for fn_name, resp in responses.items()
    }

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    if fn_name not in self._responses:
      raise AssertionError(f"unexpected rpc call: {fn_name}")
    return self._responses[fn_name]


def _make_client(
    responses: dict[str, object],
) -> sdk_client.BehaviourClient:
  fake = _DispatchingRpcClient(responses)
  return sdk_client.BehaviourClient(lambda: fake)  # type: ignore[arg-type]


def _ticket_status(
    status: rpc_api.TicketStatus,
    termination_reason: str | None = None,
    error_message: str | None = None,
) -> rpc_api.TicketStatusResponse:
  return rpc_api.TicketStatusResponse(
      info=rpc_api.TicketInfo(
          ticket_id="t1",
          status=status,
          behaviour_type="trajectory_motion",
          created_at=0.0,
          termination_reason=termination_reason,
          error_message=error_message,
      )
  )


def test_failed_ticket_raises_behaviour_failed_error() -> None:
  client = _make_client(
      {
          _INITIATE_FN: rpc_api.BehaviourInitiatedResponse(
              ticket_id="t1", error="trajectory not found: typo"
          ),
          _STATUS_FN: _ticket_status(
              rpc_api.TicketStatus.FAILED,
              termination_reason="INVALID_INPUT",
              error_message="trajectory not found: typo",
          ),
      }
  )
  future = client.trajectory_motion(trajectory_name="typo")
  with pytest.raises(sdk_client.BehaviourFailedError) as exc_info:
    future.result()
  exc = exc_info.value
  assert exc.ticket_id == "t1"
  assert exc.termination_reason == "INVALID_INPUT"
  assert "trajectory not found" in str(exc)


def test_completed_ticket_returns_normally() -> None:
  client = _make_client(
      {
          _INITIATE_FN: rpc_api.BehaviourInitiatedResponse(ticket_id="t1"),
          _STATUS_FN: _ticket_status(rpc_api.TicketStatus.COMPLETED),
      }
  )
  future = client.trajectory_motion(trajectory_name="rest")
  result = future.result()
  assert result.info is not None
  assert result.info.status is rpc_api.TicketStatus.COMPLETED


def test_initiate_error_without_ticket_raises_runtime_error() -> None:
  # Only the initiate fn is registered: an erroneous status poll would trip
  # the fake's AssertionError, catching a regression.
  client = _make_client(
      {
          _INITIATE_FN: rpc_api.BehaviourInitiatedResponse(
              ticket_id="", error="server rejected request"
          ),
      }
  )
  future = client.trajectory_motion(trajectory_name="rest")
  with pytest.raises(RuntimeError) as exc_info:
    future.result()
  exc = exc_info.value
  assert not isinstance(exc, sdk_client.BehaviourFailedError)
  assert "server rejected request" in str(exc)


def test_wait_for_ticket_raises_on_failed_state() -> None:
  client = _make_client(
      {
          _STATUS_FN: _ticket_status(
              rpc_api.TicketStatus.FAILED,
              termination_reason="INVALID_INPUT",
              error_message="trajectory not found: typo",
          ),
      }
  )
  with pytest.raises(sdk_client.BehaviourFailedError) as exc_info:
    client.wait_for_ticket("t1")
  assert exc_info.value.error_message == "trajectory not found: typo"
