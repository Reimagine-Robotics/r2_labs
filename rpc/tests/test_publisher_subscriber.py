"""Tests for the reusable pub/sub layer."""

import asyncio
import socket
import threading
import time

import pytest
import zmq

from r2_labs.rpc import publisher, subscriber


def _free_port() -> int:
  """Return a currently-free TCP port (racy but fine for tests)."""
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(("", 0))
  port = s.getsockname()[1]
  s.close()
  return port


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return False


@pytest.fixture
def running_publisher():
  events_port = _free_port()
  camera_port = _free_port()
  pub = publisher.BasePublisher(
      [
          publisher.PubSocket(
              "events", events_port, "event.", publisher.NEVER_DROP_SNDHWM
          ),
          publisher.PubSocket(
              "camera", camera_port, "camera.", publisher.LATEST_WINS_SNDHWM
          ),
      ]
  )
  pub.start()
  yield pub, events_port, camera_port
  pub.stop()


def _subscribed_on(
    pub: publisher.BasePublisher, port: int, prefix: str
) -> "subscriber.BaseSubscriber":
  """Build a subscriber and wait until the publisher sees its subscription.

  Waiting on `has_subscribers` clears the PUB/SUB slow-joiner problem: once the
  XPUB has registered the subscription, published messages will reach it.
  """
  name = next(s.name for s in pub._sockets if s.topic_prefix == prefix)
  sub = subscriber.BaseSubscriber(f"tcp://127.0.0.1:{port}")
  sub.subscribe(prefix)
  assert _wait_until(lambda: pub.has_subscribers(name))
  return sub


def _subscribed(
    pub: publisher.BasePublisher, name: str
) -> "subscriber.BaseSubscriber":
  """Subscribe to the named socket of the two-socket running_publisher fixture."""
  port = {"events": pub._sockets[0].port, "camera": pub._sockets[1].port}[name]
  prefix = {"events": "event.", "camera": "camera."}[name]
  return _subscribed_on(pub, port, prefix)


def test_publish_subscribe_roundtrip(running_publisher):
  pub, _, _ = running_publisher
  sub = _subscribed(pub, "events")
  try:
    pub.publish("event.cuff_press", b"payload")
    assert sub.recv(timeout_ms=2000) == ("event.cuff_press", b"payload")
  finally:
    sub.close()


def test_topic_prefix_routes_to_the_matching_socket(running_publisher):
  pub, _, _ = running_publisher
  # A subscriber on the camera socket must see camera messages and never an
  # event — proving the running_publisher routes event.* away from the camera socket.
  sub = _subscribed(pub, "camera")
  try:
    pub.publish("event.cuff_press", b"nope")
    pub.publish("camera.wrist", b"frame")
    assert sub.recv(timeout_ms=2000) == ("camera.wrist", b"frame")
    assert sub.recv(timeout_ms=200) is None
  finally:
    sub.close()


def test_concurrent_producer_threads(running_publisher):
  pub, _, _ = running_publisher
  sub = _subscribed(pub, "events")
  try:

    def produce(i: int) -> None:
      pub.publish("event.tick", str(i).encode())

    threads = [threading.Thread(target=produce, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    received = set()
    for _ in range(20):
      msg = sub.recv(timeout_ms=2000)
      assert msg is not None
      received.add(msg[1])
    assert received == {str(i).encode() for i in range(20)}
  finally:
    sub.close()


def test_has_subscribers_tracks_presence(running_publisher):
  pub, camera_port = running_publisher[0], running_publisher[2]
  assert not pub.has_subscribers("camera")
  sub = subscriber.BaseSubscriber(f"tcp://127.0.0.1:{camera_port}")
  sub.subscribe("camera.")
  assert _wait_until(lambda: pub.has_subscribers("camera"))
  sub.unsubscribe("camera.")
  assert _wait_until(lambda: not pub.has_subscribers("camera"))
  sub.close()


def test_start_raises_on_bind_failure_instead_of_hanging():
  port = _free_port()
  first = publisher.BasePublisher([publisher.PubSocket("a", port, "a.")])
  first.start()
  try:
    second = publisher.BasePublisher([publisher.PubSocket("b", port, "b.")])
    with pytest.raises(zmq.ZMQError):
      second.start()
  finally:
    first.stop()


def test_ambiguous_prefixes_rejected_at_construction():
  with pytest.raises(ValueError):
    publisher.BasePublisher(
        [
            publisher.PubSocket("a", _free_port(), "event."),
            publisher.PubSocket("b", _free_port(), "event.sub."),
        ]
    )


def test_double_start_is_rejected(running_publisher):
  pub = running_publisher[0]
  with pytest.raises(RuntimeError):
    pub.start()


def test_duplicate_socket_names_rejected_at_construction():
  with pytest.raises(ValueError):
    publisher.BasePublisher(
        [
            publisher.PubSocket("camera", _free_port(), "event."),
            publisher.PubSocket("camera", _free_port(), "camera."),
        ]
    )


def test_start_after_bind_failure_clears_for_retry():
  # A bind failure leaves the publisher reusable: once the port frees up, a
  # fresh start() must surface the new outcome, not re-raise the stale error.
  busy = publisher.BasePublisher([publisher.PubSocket("a", _free_port(), "a.")])
  busy.start()
  port = busy._sockets[0].port
  pub = publisher.BasePublisher([publisher.PubSocket("a", port, "a.")])
  with pytest.raises(zmq.ZMQError):
    pub.start()
  busy.stop()  # frees the port; the retry below must now bind cleanly
  pub.start()
  try:
    sub = _subscribed_on(pub, port, "a.")
    pub.publish("a.tick", b"after-retry")
    assert sub.recv(timeout_ms=2000) == ("a.tick", b"after-retry")
    sub.close()
  finally:
    pub.stop()


def test_stop_before_start_does_not_kill_publisher():
  # stop() in an error-cleanup path before start() must not leave the owner
  # loop primed to exit immediately once it finally comes up.
  port = _free_port()
  pub = publisher.BasePublisher([publisher.PubSocket("a", port, "a.")])
  pub.stop()
  pub.start()
  try:
    sub = _subscribed_on(pub, port, "a.")
    pub.publish("a.tick", b"alive")
    assert sub.recv(timeout_ms=2000) == ("a.tick", b"alive")
    sub.close()
  finally:
    pub.stop()


def test_restart_zeroes_stale_presence():
  # Stopping while a subscriber is still connected closes the XPUB before its
  # unsubscribe frame arrives, so presence stays > 0. A restart must zero it
  # rather than inherit (and accumulate) the stale count.
  port = _free_port()
  pub = publisher.BasePublisher([publisher.PubSocket("a", port, "a.")])
  pub.start()
  sub = _subscribed_on(pub, port, "a.")
  assert pub.has_subscribers("a")
  pub.stop()  # closes the XPUB before the subscriber unsubscribes
  sub.close()
  assert pub.has_subscribers("a")  # the stale count survives the stop

  pub.start()
  try:
    assert not pub.has_subscribers("a")
  finally:
    pub.stop()


def test_async_subscriber_roundtrip(running_publisher):
  pub, events_port, _ = running_publisher

  async def body() -> tuple[str, bytes]:
    sub = subscriber.AsyncBaseSubscriber(f"tcp://127.0.0.1:{events_port}")
    sub.subscribe("event.")
    assert _wait_until(lambda: pub.has_subscribers("events"))
    pub.publish("event.cuff_press", b"async")
    try:
      return await asyncio.wait_for(sub.recv(), timeout=2.0)
    finally:
      sub.close()

  assert asyncio.run(body()) == ("event.cuff_press", b"async")


def test_async_subscriber_spans_multiple_endpoints(running_publisher):
  # The streamer's pattern: one async SUB across both ports, one await loop.
  pub, events_port, camera_port = running_publisher

  async def body() -> dict[str, bytes]:
    sub = subscriber.AsyncBaseSubscriber(
        [
            f"tcp://127.0.0.1:{events_port}",
            f"tcp://127.0.0.1:{camera_port}",
        ]
    )
    sub.subscribe("event.")
    sub.subscribe("camera.")
    assert _wait_until(
        lambda: pub.has_subscribers("events") and pub.has_subscribers("camera")
    )
    pub.publish("camera.wrist", b"frame")
    pub.publish("event.cuff_press", b"press")
    received: dict[str, bytes] = {}
    try:
      for _ in range(2):
        topic, payload = await asyncio.wait_for(sub.recv(), timeout=2.0)
        received[topic] = payload
    finally:
      sub.close()
    return received

  assert asyncio.run(body()) == {
      "camera.wrist": b"frame",
      "event.cuff_press": b"press",
  }


def test_unknown_topic_is_dropped_not_fatal(running_publisher):
  pub, _, _ = running_publisher
  sub = _subscribed(pub, "events")
  try:
    pub.publish("unknown.topic", b"x")  # no matching socket: dropped, no crash
    pub.publish("event.cuff_press", b"ok")  # running_publisher still alive
    assert sub.recv(timeout_ms=2000) == ("event.cuff_press", b"ok")
  finally:
    sub.close()
