"""Consumer half of the point-to-point state stream (`DEALER`).

Receives current state from a `stream_server.BaseStreamServer`, which is where
the transport and its rationale are described. Imports the wire contract from
that module, as `client` does from `server`.

Threading. ZMQ sockets are owned by exactly one thread, so this half runs a
single owner thread that both receives frames and sends the periodic
announcement; there is deliberately no second thread for the keepalive.
"""

import os
import socket
import threading
import time
import uuid
from typing import Callable

import zmq
from loguru import logger as log

from r2_labs.rpc import stream_server

# How often a consumer announces itself. Also the granularity at which a
# producer learns this consumer still exists, and so what
# `stream_server.PEER_TIMEOUT_SEC` is sized against.
_ANNOUNCE_INTERVAL_SEC = 0.5

# How long a consumer tolerates silence before declaring the stream stale.
# Five missed announcements: three flaps on the `.local`/WiFi links this
# transport exists to span, where a GC pause or a roam can swallow a couple.
DEFAULT_STALE_AFTER_SEC = 2.5

# A `DEALER` with no peer queues rather than drops, so this bounds the backlog
# built while the producer is down. Announcements are sent non-blocking.
_SNDHWM = 4


class BaseStreamClient:
  """Receives current state from a `BaseStreamServer`.

  `on_snapshot` is called for every frame, on the client's owner thread.
  `on_stale` is called once when the stream falls silent for longer than
  `stale_after_sec`, so the consumer can mark its cached state unknown rather
  than keep reporting a value that may have changed unseen.
  """

  def __init__(
      self,
      address: str,
      on_snapshot: Callable[[bytes], None],
      on_stale: Callable[[], None],
      stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
      context: zmq.Context | None = None,
  ):
    if context is None:
      # Process-wide, as `BaseClient` does and for the same reason: a private
      # context owns its sockets and blocks in __del__ until they close.
      context = zmq.Context.instance()
    self._address = address
    self._on_snapshot = on_snapshot
    self._on_stale = on_stale
    self._stale_after_sec = stale_after_sec

    self._socket = context.socket(zmq.DEALER)
    # An explicit identity is load-bearing. Left to ZMQ, a reconnecting DEALER
    # is assigned a fresh one, so the producer keeps sending to the identity
    # that died and this socket receives nothing until its next announcement.
    # A stable identity is re-presented during the reconnect handshake, below
    # the application, so delivery resumes with no gap. Unique per instance so
    # two consumers (a probe alongside the real one) never displace each other.
    identity = (
        f"stream-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    self._socket.setsockopt(zmq.IDENTITY, identity.encode())
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.setsockopt(zmq.SNDHWM, _SNDHWM)
    stream_server.apply_tcp_keepalive(self._socket)
    self._socket.connect(address)

    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    self._is_stale = True

  @property
  def is_stale(self) -> bool:
    """Whether the stream has fallen silent. True until the first frame."""
    return self._is_stale

  def start(self) -> None:
    """Start the owner thread (idempotent)."""
    if self._thread is not None:
      return
    self._stop_event.clear()
    self._thread = threading.Thread(
        target=self._run, daemon=True, name="BaseStreamClient"
    )
    self._thread.start()

  def stop(self) -> None:
    """Stop the owner thread and wait for it to close the socket.

    Mirrors `BaseStreamServer.stop`: the owner thread closes the socket, since
    closing one under a parked poll is undefined behaviour in libzmq rather
    than an exception. A client that was never started has no owner thread to
    do it, so this closes it directly — leaving it open would strand a socket
    on the context and block a later garbage collection.
    """
    self._stop_event.set()
    thread = self._thread
    self._thread = None
    if thread is None:
      self._socket.close()
      return
    thread.join(timeout=stream_server.STOP_JOIN_TIMEOUT_SEC)
    if thread.is_alive():
      log.error(
          "stream: owner thread did not stop within {}s; leaking its socket "
          "rather than closing it underneath a live reader",
          stream_server.STOP_JOIN_TIMEOUT_SEC,
      )

  def _run(self) -> None:
    try:
      self._receive_until_stopped()
    finally:
      # This thread owns the socket, so this thread closes it.
      self._socket.close()

  def _receive_until_stopped(self) -> None:
    poller = zmq.Poller()
    poller.register(self._socket, zmq.POLLIN)
    # Announce immediately rather than after one interval, so the initial seed
    # is not delayed by the keepalive cadence.
    self._announce()
    last_announce = time.monotonic()
    last_recv = time.monotonic()
    while not self._stop_event.is_set():
      if dict(poller.poll(int(_ANNOUNCE_INTERVAL_SEC * 1000))):
        # `_drain` calls `on_snapshot` synchronously, so a consumer can read
        # `is_stale` mid-call. Clear it first, or that consumer sees its own
        # state arrive on a stream still marked stale.
        if self._is_stale:
          self._is_stale = False
          log.info("stream: {} live", self._address)
        self._drain()
        # Timestamped after delivery: a consumer slower than `stale_after_sec`
        # is a slow consumer, not a silent stream.
        last_recv = time.monotonic()
      now = time.monotonic()
      if now - last_announce >= _ANNOUNCE_INTERVAL_SEC:
        # Sent unconditionally: after a reconnect this is what re-registers
        # this consumer with the producer.
        self._announce()
        last_announce = now
      if not self._is_stale and now - last_recv >= self._stale_after_sec:
        self._is_stale = True
        log.warning("stream: {} went silent", self._address)
        self._notify_stale()

  def _drain(self) -> None:
    """Apply every frame already queued, not just the one that woke us."""
    while True:
      try:
        frames = self._socket.recv_multipart(zmq.NOBLOCK)
      except zmq.Again:
        return
      if not frames:
        continue
      self._deliver(frames[-1])

  def _deliver(self, payload: bytes) -> None:
    # A raising consumer must not kill the owner thread and take the stream
    # down with it; the next frame is delivered regardless.
    try:
      self._on_snapshot(payload)
    except Exception:  # pylint: disable=broad-except
      log.exception("stream: on_snapshot raised; continuing")

  def _notify_stale(self) -> None:
    try:
      self._on_stale()
    except Exception:  # pylint: disable=broad-except
      log.exception("stream: on_stale raised; continuing")

  def _announce(self) -> None:
    # Non-blocking: a DEALER with no live peer queues rather than drops, and a
    # blocking send would park the owner thread for the whole outage.
    try:
      self._socket.send(stream_server.READY, zmq.NOBLOCK)
    except zmq.Again:
      pass
