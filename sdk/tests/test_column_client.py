"""Tests for the SDK column client's motion futures.

Pins down what cancelling a column motion does and, as importantly, what it
cannot tell you. ArmExecutor never marks its futures running, so cancel()
always succeeds and _run_task discards whatever the task goes on to produce --
which means no failure the cancel path discovers can reach result(). A caller
who needs to know whether the column stopped calls cancel_ticket, which
answers synchronously.
"""

import pickle
import threading

import pytest

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api

_GO_TO_FN = "column.go_to"
_STATUS_FN = "behaviour.ticket_status"
_CANCEL_FN = "column.cancel_ticket"

_TICKET_ID = "tkt_000001"
_STARTED_TIMEOUT_S = 5.0


class _FakeRpcClient:
  """A column motion that keeps running until the test lets it finish."""

  def __init__(
      self,
      cancel_reply: rpc_api.CancelTicketResponse | None = None,
      cancel_raises: Exception | None = None,
  ) -> None:
    self.cancel_reply = cancel_reply or rpc_api.CancelTicketResponse(
        success=True
    )
    self.cancel_raises = cancel_raises
    self.cancelled_ticket_ids: list[str] = []
    self.polling = threading.Event()
    self._finished = threading.Event()

  def finish(self) -> None:
    """Let the motion complete, so a waiting future resolves."""
    self._finished.set()

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    del timeout
    if fn_name == _GO_TO_FN:
      return pickle.dumps(
          rpc_api.ColumnMotionInitiatedResponse(ticket_id=_TICKET_ID)
      )
    if fn_name == _STATUS_FN:
      self.polling.set()
      status = (
          rpc_api.TicketStatus.COMPLETED
          if self._finished.is_set()
          else rpc_api.TicketStatus.RUNNING
      )
      info = rpc_api.TicketInfo(
          ticket_id=_TICKET_ID,
          status=status,
          behaviour_type="column.go_to",
          created_at=0.0,
      )
      return pickle.dumps(rpc_api.TicketStatusResponse(info=info))
    if fn_name == _CANCEL_FN:
      query = pickle.loads(data) if data is not None else None
      assert isinstance(query, rpc_api.CancelTicketQuery)
      self.cancelled_ticket_ids.append(query.ticket_id)
      if self.cancel_raises is not None:
        raise self.cancel_raises
      return pickle.dumps(self.cancel_reply)
    raise AssertionError(f"unexpected rpc call: {fn_name}")


def _column_client(fake: _FakeRpcClient) -> sdk_client.ColumnClient:
  behaviour = sdk_client.BehaviourClient(lambda: fake)  # type: ignore[arg-type]
  return sdk_client.ColumnClient(fake, behaviour)  # type: ignore[arg-type]


def test_cancelling_a_started_motion_abandons_that_ticket() -> None:
  """The cancel names its own ticket, so the service can scope the stop."""
  fake = _FakeRpcClient()
  column = _column_client(fake)

  future = column.go_to(250.0)
  assert fake.polling.wait(_STARTED_TIMEOUT_S), "the motion never started"
  future.cancel()
  fake.finish()

  assert fake.cancelled_ticket_ids == [_TICKET_ID]


def test_cancelling_a_queued_motion_issues_no_cancel_at_all() -> None:
  """No ticket exists yet, so there is nothing of its own to abandon -- and
  nothing that could stop a motion belonging to somebody else."""
  fake = _FakeRpcClient()
  column = _column_client(fake)

  future = column.go_to(250.0)
  future.cancel()
  fake.finish()

  assert not fake.cancelled_ticket_ids


def test_a_refused_cancel_does_not_reach_the_caller() -> None:
  """Documents a known limit rather than an intention.

  The service refusing the stop -- the column still travelling -- is invisible
  through the future: it resolves as cancelled either way. cancel_ticket is
  the call that reports this, and the one to use when it matters.
  """
  fake = _FakeRpcClient(
      cancel_reply=rpc_api.CancelTicketResponse(
          success=False, error="the column did not answer stop"
      )
  )
  column = _column_client(fake)

  future = column.go_to(250.0)
  assert fake.polling.wait(_STARTED_TIMEOUT_S), "the motion never started"
  future.cancel()
  fake.finish()

  with pytest.raises(Exception):  # CancelledError, carrying no reason
    future.result()

  # The same refusal, asked for directly, is plainly visible.
  reply = column.cancel_ticket(_TICKET_ID)
  assert not reply.success
  assert reply.error == "the column did not answer stop"
