"""Client response and timeout behavior against the synchronous server."""

import concurrent.futures
import pickle
import threading
from collections.abc import Iterator

import pytest
import zmq

from r2_labs.rpc import client as rpc_client
from r2_labs.rpc import server as rpc_server


@pytest.fixture
def endpoint() -> Iterator[str]:
  ready: concurrent.futures.Future[rpc_server.BaseServer] = (
      concurrent.futures.Future()
  )

  def echo(data: bytes) -> bytes:
    return data

  def serve() -> None:
    with zmq.Context() as context:
      server = rpc_server.BaseServer(0, context)
      server.register_fn(echo)
      ready.set_result(server)
      server.run()

  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    serving = executor.submit(serve)
    server = ready.result(timeout=5.0)
    try:
      yield f"tcp://127.0.0.1:{server.port}"
    finally:
      server.stop()
      serving.result(timeout=5.0)


@pytest.mark.parametrize("compressed", [False, True])
def test_empty_result_is_returned_as_bytes(
    endpoint: str, compressed: bool
) -> None:
  client = rpc_client.BaseClient(
      endpoint, timeout=2000, use_compression=compressed
  )
  assert client("echo", b"") == b""
  assert pickle.loads(client("ping"))["status"] == "ack"


@pytest.mark.parametrize("first_call_fails", [False, True])
@pytest.mark.parametrize("default_timeout", [-1, 0])
def test_timeout_override_restores_unlimited_default(
    first_call_fails: bool, default_timeout: int
) -> None:
  ready: concurrent.futures.Future[rpc_server.BaseServer] = (
      concurrent.futures.Future()
  )
  started = threading.Event()
  release = threading.Event()
  finished = threading.Event()

  def slow() -> bytes:
    started.set()
    assert release.wait(timeout=5.0)
    return pickle.dumps("finished")

  def serve() -> None:
    with zmq.Context() as context:
      server = rpc_server.BaseServer(0, context)
      server.register_fn(slow)
      ready.set_result(server)
      server.run()

  def exercise(address: str) -> bytes:
    client = rpc_client.BaseClient(
        address, timeout=default_timeout, ping_on_init=False
    )
    try:
      if first_call_fails:
        with pytest.raises(rpc_client.RpcRemoteError):
          client("unknown_handler", timeout=200)
      else:
        assert pickle.loads(client("ping", timeout=200))["status"] == "ack"
      return client("slow")
    finally:
      finished.set()

  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    serving = executor.submit(serve)
    server = ready.result(timeout=5.0)
    calling = executor.submit(exercise, f"tcp://127.0.0.1:{server.port}")
    try:
      assert started.wait(timeout=2.0)
      assert not finished.wait(timeout=0.6)
      release.set()
      assert pickle.loads(calling.result(timeout=2.0)) == "finished"
    finally:
      release.set()
      server.stop()
      serving.result(timeout=5.0)
