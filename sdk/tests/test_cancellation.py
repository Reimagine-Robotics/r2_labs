"""Tests for SIGINT / emergency cancellation of in-flight behaviours."""

# Tests inspect module-internal registry/handler state by design.
# pylint: disable=protected-access

import pickle
import signal
import threading
import time

import pytest

from r2_labs.sdk import cancellation
from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api

_INITIATE_FN = "behaviour.trajectory_motion"
_STATUS_FN = "behaviour.ticket_status"
_CANCEL_FN = "behaviour.cancel_ticket"


@pytest.fixture(autouse=True)
def _clear_registry():
  """Each test starts and ends with an empty cancellation registry."""
  with cancellation._lock:
    cancellation._callbacks.clear()
  yield
  with cancellation._lock:
    cancellation._callbacks.clear()


class _StatefulRpcClient:
  """Fake RPC client whose ticket status can change between polls.

  Records cancel calls so tests can assert a cancel RPC was issued.
  """

  def __init__(self) -> None:
    self.status: rpc_api.TicketStatus = rpc_api.TicketStatus.RUNNING
    self.cancelled_ticket_ids: list[str] = []
    self.status_polls: int = 0

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    if fn_name == _INITIATE_FN:
      return pickle.dumps(rpc_api.BehaviourInitiatedResponse(ticket_id="t1"))
    if fn_name == _STATUS_FN:
      self.status_polls += 1
      info = rpc_api.TicketInfo(
          ticket_id="t1",
          status=self.status,
          behaviour_type="trajectory_motion",
          created_at=0.0,
          termination_reason=(
              "CANCELLED"
              if self.status is rpc_api.TicketStatus.FAILED
              else None
          ),
          error_message=(
              "cancelled by client"
              if self.status is rpc_api.TicketStatus.FAILED
              else None
          ),
      )
      return pickle.dumps(rpc_api.TicketStatusResponse(info=info))
    if fn_name == _CANCEL_FN:
      query = pickle.loads(data) if data is not None else None
      assert isinstance(query, rpc_api.CancelTicketQuery)
      self.cancelled_ticket_ids.append(query.ticket_id)
      return pickle.dumps(rpc_api.CancelTicketResponse(success=True))
    raise AssertionError(f"unexpected rpc call: {fn_name}")


def test_cancel_all_inflight_fires_every_callback() -> None:
  fired: list[str] = []
  cancellation.register_cancel(lambda: fired.append("a"))
  cancellation.register_cancel(lambda: fired.append("b"))
  count = cancellation.cancel_all_inflight()
  assert count == 2
  assert sorted(fired) == ["a", "b"]


def test_cancel_all_inflight_is_best_effort() -> None:
  fired: list[str] = []

  def _boom() -> None:
    raise RuntimeError("callback failed")

  cancellation.register_cancel(_boom)
  cancellation.register_cancel(lambda: fired.append("ok"))
  # Must not propagate the callback error, and must still fire the other.
  count = cancellation.cancel_all_inflight()
  assert count == 2
  assert fired == ["ok"]


def test_unregister_removes_callback() -> None:
  fired: list[str] = []
  token = cancellation.register_cancel(lambda: fired.append("a"))
  cancellation.unregister_cancel(token)
  assert cancellation.cancel_all_inflight() == 0
  assert not fired


def test_inflight_behaviour_is_cancellable_end_to_end() -> None:
  fake = _StatefulRpcClient()
  client = sdk_client.BehaviourClient(lambda: fake)  # type: ignore[arg-type]
  future = client.trajectory_motion(trajectory_name="rest")

  # Wait until the worker has polled status at least once, which guarantees the
  # ticket id is set on the cancel callback (it is populated just before the
  # first poll).
  _wait_until(lambda: fake.status_polls >= 1)

  cancellation.cancel_all_inflight()
  assert fake.cancelled_ticket_ids == ["t1"]

  # Server now reports the cancellation as a FAILED(CANCELLED) terminal state.
  fake.status = rpc_api.TicketStatus.FAILED
  with pytest.raises(sdk_client.BehaviourFailedError) as exc_info:
    future.result()
  assert exc_info.value.termination_reason == "CANCELLED"
  # finally in _task unregisters the callback once resolved.
  _wait_until(lambda: len(cancellation._callbacks) == 0)
  assert len(cancellation._callbacks) == 0


def test_wait_for_ticket_grace_window_raises_cancelled(monkeypatch) -> None:
  monkeypatch.setattr(sdk_client, "_CANCEL_GRACE_SECONDS", 0.0)
  monkeypatch.setattr(sdk_client.time, "sleep", lambda _seconds: None)

  fake = _StatefulRpcClient()  # status stays RUNNING forever
  client = sdk_client.BehaviourClient(lambda: fake)  # type: ignore[arg-type]
  cancel_event = threading.Event()
  cancel_event.set()
  with pytest.raises(sdk_client.BehaviourCancelledError) as exc_info:
    client.wait_for_ticket("t1", cancel_event=cancel_event)
  assert exc_info.value.ticket_id == "t1"


def test_handle_sigint_chains_to_user_handler(monkeypatch) -> None:
  called: list[int] = []
  monkeypatch.setattr(
      cancellation,
      "_previous_sigint_handler",
      lambda signum, frame: called.append(signum),
  )
  # Should invoke the prior handler and NOT raise KeyboardInterrupt.
  cancellation._handle_sigint(signal.SIGINT, None)
  assert called == [signal.SIGINT]


def test_handle_sigint_default_raises_keyboard_interrupt(monkeypatch) -> None:
  monkeypatch.setattr(cancellation, "_previous_sigint_handler", signal.SIG_DFL)
  with pytest.raises(KeyboardInterrupt):
    cancellation._handle_sigint(signal.SIGINT, None)


def test_handle_sigint_ignored_does_not_raise(monkeypatch) -> None:
  monkeypatch.setattr(cancellation, "_previous_sigint_handler", signal.SIG_IGN)
  cancellation._handle_sigint(signal.SIGINT, None)  # must not raise


def test_install_handlers_off_main_thread_is_noop(monkeypatch) -> None:
  monkeypatch.setattr(cancellation, "_handler_installed", False)
  result: dict[str, bool] = {}

  def _install_in_thread() -> None:
    cancellation.install_handlers()
    result["installed"] = cancellation._handler_installed

  thread = threading.Thread(target=_install_in_thread)
  thread.start()
  thread.join()
  assert result["installed"] is False


def _wait_until(predicate, timeout: float = 2.0) -> None:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if predicate():
      return
    time.sleep(0.01)
  raise AssertionError("condition not met within timeout")
