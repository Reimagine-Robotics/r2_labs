"""The rpc metric group: both ends record series, old peers degrade cleanly."""

import pickle
import threading

import prometheus_client
import pytest
import zmq

from r2_labs.rpc import client as rpc_client
from r2_labs.rpc import server as rpc_server


def _sample(name: str, labels: dict[str, str]) -> float | None:
  return prometheus_client.REGISTRY.get_sample_value(name, labels)


def _echo(data: bytes) -> bytes:
  return data


def _boom(data: bytes) -> bytes:
  del data
  raise RuntimeError("handler failure")


@pytest.fixture(name="rpc_pair")
def rpc_pair_fixture():
  context = zmq.Context()
  server = rpc_server.BaseServer(0, context)
  server.register_fn(_echo, fn_name="echo")
  server.register_fn(_boom, fn_name="boom")
  thread = threading.Thread(target=server.run, daemon=True)
  thread.start()
  client = rpc_client.BaseClient(
      f"tcp://localhost:{server.port}", timeout=2000, service_name="test-svc"
  )
  try:
    yield server, client
  finally:
    # Stop the loop and join before destroying the context: closing sockets
    # under a parked recv aborts the process inside libzmq.
    server.stop()
    thread.join(timeout=5)
    context.destroy(linger=0)


def test_client_and_server_record_call_series(rpc_pair):
  _, client = rpc_pair
  labels = {"service": "test-svc", "fn": "echo"}
  duration_before = (
      _sample("r2_rpc_client_request_duration_seconds_count", labels) or 0
  )
  overhead_before = _sample("r2_rpc_client_overhead_seconds_count", labels) or 0
  requests_before = (
      _sample("r2_rpc_server_requests_total", {"handler": "echo"}) or 0
  )

  assert client("echo", pickle.dumps("payload")) == pickle.dumps("payload")

  assert (
      _sample("r2_rpc_client_request_duration_seconds_count", labels)
      == duration_before + 1
  )
  # The server returned its busy time, so overhead was measurable, and busy
  # time can't exceed what the client observed end to end.
  assert (
      _sample("r2_rpc_client_overhead_seconds_count", labels)
      == overhead_before + 1
  )
  assert (
      _sample("r2_rpc_server_requests_total", {"handler": "echo"})
      == requests_before + 1
  )
  assert (
      _sample("r2_rpc_handler_duration_seconds_count", {"handler": "echo"})
      is not None
  )


def test_remote_error_counted(rpc_pair):
  _, client = rpc_pair
  labels = {"service": "test-svc", "fn": "boom", "error": "remote"}
  errors_before = _sample("r2_rpc_client_errors_total", labels) or 0

  with pytest.raises(rpc_client.RpcRemoteError):
    client("boom", pickle.dumps("payload"))

  assert _sample("r2_rpc_client_errors_total", labels) == errors_before + 1


def test_timeout_counted_and_observed():
  # No server behind this address: the call times out locally.
  client_context = zmq.Context()
  error_labels = {"service": "nobody", "fn": "ping", "error": "timeout"}
  duration_labels = {"service": "nobody", "fn": "ping"}
  errors_before = _sample("r2_rpc_client_errors_total", error_labels) or 0
  durations_before = (
      _sample("r2_rpc_client_request_duration_seconds_count", duration_labels)
      or 0
  )
  try:
    with pytest.raises(rpc_client.RpcTimeoutError):
      rpc_client.BaseClient(
          "tcp://localhost:1", timeout=100, service_name="nobody"
      )
  finally:
    client_context.destroy(linger=0)
  assert (
      _sample("r2_rpc_client_errors_total", error_labels) == errors_before + 1
  )
  # Failed calls still land in the duration histogram, at the deadline.
  assert (
      _sample("r2_rpc_client_request_duration_seconds_count", duration_labels)
      == durations_before + 1
  )


def test_old_client_without_busy_time_flag_gets_plain_reply(rpc_pair):
  """An RpcArgs pickle that predates return_busy_time still round-trips."""
  server, _ = rpc_pair
  args = rpc_server.RpcArgs(fn_name="echo", fn_args=pickle.dumps("old"))
  # Simulate the old wire format: strip the new field from the pickled state,
  # the way an old client's pickle simply never carries it.
  del args.__dict__["return_busy_time"]

  context = zmq.Context()
  sock = context.socket(zmq.REQ)
  sock.setsockopt(zmq.RCVTIMEO, 2000)
  sock.connect(f"tcp://localhost:{server.port}")
  try:
    sock.send(pickle.dumps(args))
    reply = sock.recv()
    assert reply == pickle.dumps("old")
    # Single-frame reply: no busy-time envelope for a client that didn't ask.
    assert sock.getsockopt(zmq.RCVMORE) == 0
  finally:
    context.destroy(linger=0)


def test_service_label_falls_back_to_endpoint(rpc_pair):
  server, _ = rpc_pair
  client = rpc_client.BaseClient(f"tcp://localhost:{server.port}", timeout=2000)
  labels = {"service": f"localhost:{server.port}", "fn": "ping"}
  # Construction pings, so the endpoint-labelled series already exists.
  assert _sample("r2_rpc_client_request_duration_seconds_count", labels) >= 1
  del client
