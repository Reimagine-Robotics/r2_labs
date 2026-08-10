"""Producer half of the point-to-point state stream (`ROUTER`).

The third transport shape in this package, alongside `server`/`client`
(request/response) and `publisher`/`subscriber` (broadcast fan-out). Use it to
stream *current state* from one producer to a small number of consumers;
`stream_client` is the other half.

Why not pub/sub. A `PUB` socket is a broadcast: the publisher never learns who
is listening, so it cannot hand a newly-arrived consumer the current value, and
a late joiner sees nothing until the next change. Every pub/sub consumer
therefore needs a second, separate transport to seed itself, and a heartbeat
bolted on to notice the producer dying. `ROUTER` knows its peers, so both fall
out for free: a consumer announces itself and is answered with the current
state on the same socket, in order, and its keepalive doubles as the liveness
signal.

The stream carries opaque `bytes`. Mirrors `BaseServer`: construction-time
wiring, bytes in/out, typed wrappers on top.

It streams *state*, not events: delivery is latest-wins, so a value that
appears and reverts between two of the producer's wakes is never sent. See
`set_snapshot`.

Threading. ZMQ sockets are owned by exactly one thread, so this half runs a
single owner thread and every public method is a thread-safe handoff to it.
Producers call `set_snapshot` from any thread; it takes a lock and sets a flag,
and never touches the socket.
"""

import threading
import time

import zmq
from loguru import logger as log

# Consumer -> producer announcement. A `ROUTER` can only address a peer it has
# already received from, so this is what makes a consumer reachable at all;
# the rest is opportunistic reuse of that mandatory first message. It doubles
# as "send me the current state", which is what re-seeds a consumer after a
# reconnect, and as the keepalive that keeps the peer from being pruned.
# `READY` follows the ZMQ guide's worker-registration convention.
READY = b"READY"

# How long a producer keeps sending to a peer that has stopped announcing
# itself. Generous relative to the client's keepalive cadence so a stalled
# consumer is not dropped for one late message, tight enough that dead peers
# (every healthcheck probe connects as a new one) do not accumulate.
PEER_TIMEOUT_SEC = 5.0

# Upper bound on how long stop() waits for an owner thread to notice. Shared
# with `stream_client`, whose half has the same shutdown shape.
STOP_JOIN_TIMEOUT_SEC = 2.0

# Producer loop wake interval, and so the upper bound on how long a state
# change waits before being sent. Small enough to be imperceptible; the cost is
# a trivial syscall that returns immediately. A producer needing true
# zero-latency wake would add an inproc pipe to the poller, as `publisher` does.
_POLL_INTERVAL_MS = 5

# Bounds memory against a peer that has wedged but not yet timed out. Dropping
# is correct here: every message is a complete snapshot, so the next one
# supersedes anything lost.
_SNDHWM = 8


def apply_tcp_keepalive(sock: zmq.Socket) -> None:
  """Detect a peer that vanished without closing the connection.

  A yanked cable or a dropped WiFi link leaves the other end writing into a
  pipe nobody reads. Without this the OS default (~2 hours) decides when that
  is noticed. Shared with `stream_client`, since both ends need it.
  """
  sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
  sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
  sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
  sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)


class BaseStreamServer:
  """Serves the current state to every connected consumer.

  Call `set_snapshot` whenever the state changes; connected consumers receive
  it immediately, and a consumer arriving later is answered with the latest one
  as soon as it announces itself.
  """

  def __init__(self, port: int, context: zmq.Context | None = None):
    if context is None:
      # Process-wide, as `BaseClient` does and for the same reason: a private
      # context owns its sockets and blocks in __del__ until they close. This
      # half has no inproc plumbing, so it needs no context of its own.
      context = zmq.Context.instance()
    self._socket = context.socket(zmq.ROUTER)
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.setsockopt(zmq.SNDHWM, _SNDHWM)
    # A reconnecting consumer presents the identity it used before. Without
    # handover the ROUTER rejects it until the stale pipe is reaped, which on
    # a hard disconnect can take the TCP keepalive timeout.
    self._socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
    apply_tcp_keepalive(self._socket)
    self._socket.bind(f"tcp://*:{port}")
    # Actual bound port, so a caller passing 0 can discover the ephemeral one
    # without a racy pick-a-free-port-then-rebind dance (as `BaseServer` does).
    endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
    self.port = int(endpoint.rsplit(":", 1)[1])

    self._lock = threading.Lock()
    self._snapshot: bytes | None = None
    self._dirty = threading.Event()
    # Identity -> when it last announced itself, so peers that stopped talking
    # are pruned instead of accumulating for the life of the process.
    self._peers: dict[bytes, float] = {}
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None

  def set_snapshot(self, payload: bytes) -> None:
    """Publish `payload` as the current state. Safe from any thread.

    Latest-wins: a snapshot set before the owner thread has sent the previous
    one replaces it rather than queueing behind it. Because every payload is
    complete state rather than a delta, the value consumers converge on is
    always correct, and two changes that arrive together are sent as one
    correct snapshot rather than two.

    What that costs is transients: a value that appears and reverts within one
    `_POLL_INTERVAL_MS` wake is never sent at all. For state that is the right
    trade — and it absorbs contact bounce for free — but it makes this the
    wrong transport for a consumer that must observe every individual event.
    """
    with self._lock:
      self._snapshot = payload
    self._dirty.set()

  def start(self) -> None:
    """Start the owner thread (idempotent)."""
    if self._thread is not None:
      return
    self._stop_event.clear()
    self._thread = threading.Thread(
        target=self._run, daemon=True, name="BaseStreamServer"
    )
    self._thread.start()

  def stop(self) -> None:
    """Stop the owner thread and wait for it to close the socket.

    The socket is closed by the owner thread on its way out, not from here.
    Closing it under a parked `poll`/`recv` is undefined behaviour in libzmq
    rather than an exception, so a bounded join that expired must leave the
    socket alone and say so.
    """
    self._stop_event.set()
    thread = self._thread
    self._thread = None
    if thread is None:
      # Never started, so the owner thread will not close it for us.
      self._socket.close()
      return
    thread.join(timeout=STOP_JOIN_TIMEOUT_SEC)
    if thread.is_alive():
      log.error(
          "stream: owner thread did not stop within {}s; leaking its socket "
          "rather than closing it underneath a live reader",
          STOP_JOIN_TIMEOUT_SEC,
      )

  def _run(self) -> None:
    poller = zmq.Poller()
    poller.register(self._socket, zmq.POLLIN)
    try:
      while not self._stop_event.is_set():
        # Drain every pending announcement before considering a state change,
        # so a consumer that has just arrived is registered in time to receive
        # it.
        if dict(poller.poll(_POLL_INTERVAL_MS)):
          self._handle_announcements()
        # Prune every pass, not only when broadcasting: an idle producer never
        # broadcasts, and healthcheck probes keep arriving as fresh identities.
        self._prune_departed_peers()
        if self._dirty.is_set():
          self._dirty.clear()
          self._broadcast(self._current_snapshot())
    finally:
      # This thread owns the socket, so this thread closes it.
      self._socket.close()

  def _handle_announcements(self) -> None:
    snapshot = self._current_snapshot()
    while True:
      try:
        frames = self._socket.recv_multipart(zmq.NOBLOCK)
      except zmq.Again:
        return
      if len(frames) < 2 or frames[1] != READY:
        log.warning("stream: ignoring unrecognised frame from peer")
        continue
      identity = frames[0]
      self._peers[identity] = time.monotonic()
      # Answer immediately, so a consumer is seeded on connect rather than
      # waiting for the next state change. Taken after any snapshot already
      # sent to this peer, so it can never deliver older state than they hold.
      if snapshot is not None:
        self._send(identity, snapshot)

  def _prune_departed_peers(self) -> None:
    """Forget peers that have stopped announcing themselves."""
    cutoff = time.monotonic() - PEER_TIMEOUT_SEC
    for identity, last_seen in list(self._peers.items()):
      if last_seen < cutoff:
        del self._peers[identity]

  def _broadcast(self, snapshot: bytes | None) -> None:
    if snapshot is None:
      return
    for identity in list(self._peers):
      self._send(identity, snapshot)

  def _send(self, identity: bytes, snapshot: bytes) -> None:
    # Never blocks: unroutable identities and peers at their high-water mark
    # are dropped by ZMQ, and the next snapshot supersedes anything lost.
    try:
      self._socket.send_multipart([identity, snapshot], zmq.NOBLOCK)
    except zmq.ZMQError as e:
      log.debug("stream: dropping frame for peer: {}", e)

  def _current_snapshot(self) -> bytes | None:
    with self._lock:
      return self._snapshot
