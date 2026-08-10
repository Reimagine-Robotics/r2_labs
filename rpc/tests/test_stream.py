"""Tests for the point-to-point state stream.

The failure modes worth pinning here are the ones a hardware session cannot
reach: seeding a consumer that arrives late, surviving a reconnect, pruning
peers that stopped talking, and reporting silence as staleness.
"""

import threading
import time

import pytest
import zmq

from r2_labs.rpc import stream_client, stream_server


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return False


class _Sink:
  """Records snapshots and stale transitions from the client's owner thread."""

  def __init__(self) -> None:
    self.snapshots: list[bytes] = []
    self.stale_count = 0
    self._lock = threading.Lock()

  def on_snapshot(self, payload: bytes) -> None:
    with self._lock:
      self.snapshots.append(payload)

  def on_stale(self) -> None:
    with self._lock:
      self.stale_count += 1

  @property
  def last(self) -> bytes | None:
    with self._lock:
      return self.snapshots[-1] if self.snapshots else None

  @property
  def count(self) -> int:
    with self._lock:
      return len(self.snapshots)


@pytest.fixture(name="context")
def _context():
  ctx = zmq.Context()
  yield ctx
  ctx.term()


@pytest.fixture(name="server")
def _server(context):
  # Port 0 binds an ephemeral port; `.port` reports the one actually bound.
  server = stream_server.BaseStreamServer(port=0, context=context)
  server.start()
  yield server
  server.stop()


def _client(context, server, sink, **kwargs) -> stream_client.BaseStreamClient:
  return stream_client.BaseStreamClient(
      address=f"tcp://127.0.0.1:{server.port}",
      on_snapshot=sink.on_snapshot,
      on_stale=sink.on_stale,
      context=context,
      **kwargs,
  )


def test_late_consumer_is_seeded_with_current_state(context, server):
  """A consumer that connects after the last change still learns it.

  This is the property pub/sub cannot offer and the reason this transport
  exists: the state was set before the consumer existed.
  """
  server.set_snapshot(b"already-happened")

  sink = _Sink()
  client = _client(context, server, sink)
  client.start()
  try:
    assert _wait_until(lambda: sink.last == b"already-happened")
  finally:
    client.stop()


def test_changes_are_pushed_without_being_asked_for(context, server):
  sink = _Sink()
  client = _client(context, server, sink)
  client.start()
  try:
    server.set_snapshot(b"first")
    assert _wait_until(lambda: sink.last == b"first")
    server.set_snapshot(b"second")
    assert _wait_until(lambda: sink.last == b"second")
  finally:
    client.stop()


def test_every_transition_arrives(context, server):
  """No sampling gap: a burst of changes is delivered, not coalesced away."""
  sink = _Sink()
  client = _client(context, server, sink)
  client.start()
  try:
    # Seed first: a producer with no state yet has nothing to answer an announcement
    # with, so wait for the consumer to be live before counting transitions.
    server.set_snapshot(b"seed")
    assert _wait_until(lambda: sink.count > 0)
    baseline = sink.count
    for i in range(20):
      server.set_snapshot(f"tick-{i}".encode())
      # Space the changes past the producer's wake interval so each is a
      # distinct send rather than being folded into one by the dirty flag.
      time.sleep(0.02)
    assert _wait_until(lambda: sink.last == b"tick-19")
    assert sink.count - baseline >= 20
  finally:
    client.stop()


def test_rapid_changes_coalesce_to_the_latest_state(context, server):
  """Delivery is latest-wins, and that is deliberate.

  Two changes set back-to-back are sent as one snapshot, not two. Because each
  payload is complete state rather than a delta, the consumer still converges
  on the correct value; what is lost is only the intermediate one. Pinned here
  so the behaviour is a decision rather than an accident — turning this into a
  queue would change the transport's contract.
  """
  sink = _Sink()
  client = _client(context, server, sink)
  client.start()
  try:
    server.set_snapshot(b"seed")
    assert _wait_until(lambda: sink.last == b"seed")
    server.set_snapshot(b"intermediate")
    server.set_snapshot(b"final")
    assert _wait_until(lambda: sink.last == b"final")
    # The intermediate value never went out. Deliberately not asserting an
    # exact frame count: the producer also re-answers keepalives, so one
    # landing in this window is legitimate and would make that flaky.
    assert b"intermediate" not in sink.snapshots
  finally:
    client.stop()


def test_silence_is_reported_as_stale(context, server):
  sink = _Sink()
  client = _client(context, server, sink, stale_after_sec=0.5)
  client.start()
  try:
    server.set_snapshot(b"alive")
    # A frame having arrived is what proves the stream was live; `is_stale` is
    # deliberately not asserted here, because the client thread owns it and
    # would legitimately flip it after `stale_after_sec` while this thread was
    # descheduled.
    assert _wait_until(lambda: sink.last == b"alive")
    # Killing the producer's owner thread stops both pushes and announcement
    # replies.
    server.stop()
    assert _wait_until(lambda: client.is_stale, timeout_s=3.0)
    assert sink.stale_count == 1
  finally:
    client.stop()


def test_peers_that_stop_announcing_are_pruned(context, server):
  sink = _Sink()
  client = _client(context, server, sink)
  client.start()
  try:
    server.set_snapshot(b"x")
    assert _wait_until(lambda: sink.last == b"x")
    assert len(server._peers) == 1  # pylint: disable=protected-access
  finally:
    client.stop()

  # With the consumer gone its identity stops being refreshed, so the next
  # broadcast after the timeout drops it rather than keeping it forever.
  time.sleep(stream_server.PEER_TIMEOUT_SEC + 0.2)  # pylint: disable=protected-access
  server.set_snapshot(b"y")
  assert _wait_until(
      lambda: not server._peers  # pylint: disable=protected-access
  )


def test_consumer_recovers_when_the_producer_restarts(context):
  """The consumer re-seeds itself after the producer goes away and comes back.

  This is the realistic outage — the producer's container restarts — and the
  property that makes an explicit DEALER identity load-bearing: the reconnect
  happens below the application, and the next announcement re-registers this consumer
  with the new producer. Nothing restarts the client.
  """
  first = stream_server.BaseStreamServer(port=0, context=context)
  first.start()
  port = first.port

  sink = _Sink()
  client = stream_client.BaseStreamClient(
      address=f"tcp://127.0.0.1:{port}",
      on_snapshot=sink.on_snapshot,
      on_stale=sink.on_stale,
      stale_after_sec=0.5,
      context=context,
  )
  client.start()
  try:
    first.set_snapshot(b"before")
    assert _wait_until(lambda: sink.last == b"before")

    first.stop()
    assert _wait_until(lambda: client.is_stale, timeout_s=3.0)

    second = stream_server.BaseStreamServer(port=port, context=context)
    second.start()
    try:
      second.set_snapshot(b"after")
      # No client restart, no manual reseed: delivery resumes on its own.
      assert _wait_until(lambda: sink.last == b"after", timeout_s=5.0)
      assert not client.is_stale
    finally:
      second.stop()
  finally:
    client.stop()


def test_two_consumers_both_receive(context, server):
  """A probe attaching must not displace the real consumer."""
  sink_a, sink_b = _Sink(), _Sink()
  client_a = _client(context, server, sink_a)
  client_b = _client(context, server, sink_b)
  client_a.start()
  client_b.start()
  try:
    server.set_snapshot(b"both")
    assert _wait_until(lambda: sink_a.last == b"both")
    assert _wait_until(lambda: sink_b.last == b"both")
  finally:
    client_a.stop()
    client_b.stop()


def test_a_raising_consumer_does_not_kill_the_stream(context, server):
  """One bad frame handler must not take the whole stream down."""
  seen: list[bytes] = []

  def explode(payload: bytes) -> None:
    seen.append(payload)
    raise RuntimeError("boom")

  client = stream_client.BaseStreamClient(
      address=f"tcp://127.0.0.1:{server.port}",
      on_snapshot=explode,
      on_stale=lambda: None,
      context=context,
  )
  client.start()
  try:
    server.set_snapshot(b"one")
    assert _wait_until(lambda: b"one" in seen)
    server.set_snapshot(b"two")
    assert _wait_until(lambda: b"two" in seen)
  finally:
    client.stop()
