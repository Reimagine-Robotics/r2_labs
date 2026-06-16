"""ZMQ publisher for the pub/sub broadcast plane.

`BasePublisher` fans messages from many producer threads onto one `XPUB`
socket per delivery policy. The owner thread routes by topic prefix, so
producers just tag messages with a topic and let the publisher handle the rest.
"""

import dataclasses
import threading
import uuid
from typing import Protocol

import zmq
from loguru import logger as log

# zmq's default send high water mark (SNDHWM) is 1000. "never-drop" sockets use
# a generous bound so small, infrequent control events effectively never drop;
# "latest-wins" sockets use a depth of 1 so a slow consumer skips stale messages
# rather than building a backlog.
NEVER_DROP_SNDHWM = 100_000
LATEST_WINS_SNDHWM = 1

# When a subscriber (un)subscribes, libzmq delivers a one-frame notification to
# the XPUB socket: first byte \x01 for subscribe or \x00 for unsubscribe,
# followed by the subscription topic. See the ZMQ_XPUB section of
# http://api.zeromq.org/master:zmq-socket
_SUBSCRIBE_BYTE = b"\x01"
_UNSUBSCRIBE_BYTE = b"\x00"

_POLL_TIMEOUT_MS = 200


class EventPublisher(Protocol):
  """The narrow publish-side contract an event producer depends on.

  `BasePublisher` implements it. Producers depend on this rather than the full
  publisher, so they stay decoupled from its construction/lifecycle and are
  trivial to fake in tests.
  """

  def publish(self, topic: str, payload: bytes) -> None:
    ...


@dataclasses.dataclass(frozen=True)
class PubSocket:
  """One `XPUB` socket dedicated to a single delivery policy.

  Attributes:
    name: short identifier for the socket (e.g. "events", "camera"), used for
      subscriber-presence lookups.
    port: TCP port the `XPUB` binds.
    topic_prefix: topics with this prefix route to this socket (e.g. "event.",
      "camera."). Prefixes must be unambiguous across a publisher's sockets.
    sndhwm: send high-water mark — the buffer depth past which the socket
      silently drops outgoing messages. Large = never-drop; 1 = latest-wins.
  """

  name: str
  port: int
  topic_prefix: str
  sndhwm: int = NEVER_DROP_SNDHWM


class BasePublisher(EventPublisher):
  """Owns one `XPUB` per delivery policy and fans producer threads onto them.

  ZMQ sockets are single-thread-owned, so a dedicated owner thread owns every
  `XPUB` plus the receiving end of an in-process (`inproc`) fan-in. Producers on
  any thread call `publish`, which sends over the fan-in; the owner thread
  routes each message by topic prefix to the matching `XPUB`. The owner thread
  also reads `XPUB` subscription messages, so `has_subscribers` can gate
  producers (e.g. the camera encodes only while someone is watching).

  Construct with the sockets the service needs, then call `start`. Mirrors RPC's
  `BaseServer`: construction-time wiring, bytes in/out, typed wrappers on top.
  """

  def __init__(
      self,
      sockets: list[PubSocket],
      context: zmq.Context | None = None,
  ):
    if not sockets:
      raise ValueError("BasePublisher requires at least one PubSocket")
    # Socket names must be unique, otherwise two sockets would share one
    # presence counter and has_subscribers would conflate their subscriptions.
    if len({s.name for s in sockets}) != len(sockets):
      raise ValueError("PubSocket names must be unique")
    # Routing is first-match by prefix, so no prefix may be a prefix of another
    # (an empty prefix would swallow everything). After sorting, any such pair
    # is adjacent, so checking neighbours catches all overlaps and duplicates.
    ordered = sorted(s.topic_prefix for s in sockets)
    for shorter, longer in zip(ordered, ordered[1:]):
      if longer.startswith(shorter):
        raise ValueError(
            f"Ambiguous topic prefixes: {shorter!r} is a prefix of {longer!r}"
        )
    self._sockets = sockets
    # One zmq context, shared by the owner thread and every producer thread:
    # contexts are thread-safe to share (individual sockets are not), and the
    # inproc fan-in only works between sockets in the same context.
    self._context = context if context is not None else zmq.Context()
    # inproc endpoints are scoped to a context; a unique address lets several
    # publishers coexist in one process without colliding.
    self._fanin_address = f"inproc://pubsub-fanin-{uuid.uuid4().hex}"
    self._local = threading.local()
    self._stop = threading.Event()
    self._ready = threading.Event()
    self._thread: threading.Thread | None = None
    self._presence_lock = threading.Lock()
    self._presence: dict[str, int] = {s.name: 0 for s in sockets}
    self._xpubs: list[tuple[PubSocket, zmq.Socket]] = []
    self._startup_error: Exception | None = None

  def start(self) -> None:
    """Spawn the owner thread and block until its sockets are bound.

    Binding before returning means a `publish` immediately after `start`
    cannot lose its inproc connection to an unbound fan-in. Raises if binding
    fails (e.g. a port is already in use) rather than leaving `start` hung.
    """
    if self._thread is not None:
      raise RuntimeError("BasePublisher.start() called twice")
    # Reset the lifecycle state so a retry after a failed start (or after a
    # stop() that preceded the first start) begins from a clean slate rather
    # than waking instantly on a stale _ready or re-raising a stale error.
    self._ready.clear()
    self._stop.clear()
    self._startup_error = None
    # Zero presence too: stopping while subscribers are still connected closes
    # the XPUBs before their unsubscribe frames arrive, so stale counts would
    # otherwise carry into — and accumulate across — each restart. No lock: the
    # owner thread that mutates _presence isn't running yet.
    self._presence = {s.name: 0 for s in self._sockets}
    self._thread = threading.Thread(
        target=self._run, name="pubsub-publisher", daemon=True
    )
    self._thread.start()
    self._ready.wait()
    if self._startup_error is not None:
      self._thread.join()
      self._thread = None  # allow a retry; the owner thread never came up
      raise self._startup_error

  def stop(self) -> None:
    """Signal the owner thread to exit and join it.

    Closes the owner thread's sockets. Per-thread `PUSH` sockets created by
    `publish` on other threads are not closed here (a socket can only be closed
    by its owning thread). Let the context be reclaimed at process exit rather
    than calling `context.term()`, which would block on those still-open
    sockets.
    """
    self._stop.set()
    if self._thread is not None:
      self._thread.join()
      self._thread = None  # allow a subsequent start() after a clean stop

  def publish(self, topic: str, payload: bytes) -> None:
    """Publish `payload` under `topic` from any thread.

    The calling thread lazily gets its own `PUSH` socket to the fan-in, so
    concurrent producers never share a socket or take a lock. The send can in
    principle block if the owner thread falls behind (a `PUSH` socket blocks at
    its SNDHWM), but the owner thread only routes — and `XPUB` drops rather
    than blocks at its own HWM — so it keeps up at the rates this carries.
    Must be called after `start`.
    """
    push = getattr(self._local, "push", None)
    if push is None:
      push = self._context.socket(zmq.PUSH)
      push.setsockopt(zmq.LINGER, 0)
      push.connect(self._fanin_address)
      self._local.push = push
    push.send_multipart([topic.encode(), payload])

  def has_subscribers(self, socket_name: str) -> bool:
    """Whether the named socket currently has at least one subscriber."""
    with self._presence_lock:
      return self._presence[socket_name] > 0

  def _route(self, topic: bytes) -> zmq.Socket:
    for spec, sock in self._xpubs:
      if topic.startswith(spec.topic_prefix.encode()):
        return sock
    raise KeyError(topic)

  def _run(self) -> None:
    # The owner thread's whole life: bind the sockets, then loop forwarding
    # published messages and tracking subscriber presence until stop().
    # Binding happens here, not in start(), because a ZMQ socket may be used
    # only on the thread that created it — and this thread runs the loop below.
    # start() just waits (via _ready) for this setup to finish.
    #
    # A Poller lets one thread wait on several sockets at once and ask which
    # ones have data ready, so this thread sleeps rather than busy-spinning.
    poller = zmq.Poller()
    pull: zmq.Socket | None = None
    try:
      # PULL is the inbox: the receiving end of the inproc fan-in that every
      # producer thread's publish() sends into.
      pull = self._context.socket(zmq.PULL)
      pull.setsockopt(zmq.LINGER, 0)
      pull.bind(self._fanin_address)
      poller.register(pull, zmq.POLLIN)
      # Setup one outbound XPUB per delivery policy.
      for spec in self._sockets:
        sock = self._context.socket(zmq.XPUB)
        # Track the socket before bind() so _close_sockets() reclaims it even if
        # bind fails (e.g. port in use).
        self._xpubs.append((spec, sock))
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, spec.sndhwm)
        sock.bind(f"tcp://*:{spec.port}")
        poller.register(sock, zmq.POLLIN)
    except Exception as exc:  # pylint: disable=broad-except
      # Surface bind failures (e.g. port in use) to start() instead of hanging.
      self._startup_error = exc
      self._close_sockets(pull)
      self._ready.set()
      return

    self._ready.set()  # sockets are bound; let start() return
    try:
      while not self._stop.is_set():
        # Wait for any socket to have data, but wake at least every timeout so
        # a stop() between messages is still noticed promptly.
        ready = dict(poller.poll(timeout=_POLL_TIMEOUT_MS))
        if pull in ready:
          # A producer published: forward [topic, payload] out the XPUB whose
          # prefix matches the topic — i.e. broadcast it to subscribers.
          frames = pull.recv_multipart()
          if len(frames) != 2:
            log.warning(
                "pubsub: dropping malformed message ({} frames)", len(frames)
            )
          else:
            topic, payload = frames
            try:
              self._route(topic).send_multipart([topic, payload])
            except KeyError:
              log.warning(
                  "pubsub: no socket matches topic {!r}; dropped", topic
              )
        # A subscriber subscribed/unsubscribed on one of the XPUBs: update its
        # presence.
        for spec, sock in self._xpubs:
          if sock in ready:
            self._on_subscription(spec.name, sock.recv())
    finally:
      self._close_sockets(pull)

  def _close_sockets(self, pull: zmq.Socket | None) -> None:
    if pull is not None:
      pull.close()
    for _, sock in self._xpubs:
      sock.close()
    self._xpubs = []

  def _on_subscription(self, name: str, frame: bytes) -> None:
    # Driven by XPUB sub/unsub frames. A clean unsubscribe or socket close is
    # reported; a hard peer disconnect may not be, so presence can stay > 0
    # until reconnect (bounded over-encoding, never corruption). Only the two
    # known verbs move the counter; anything else is ignored rather than
    # treated as an unsubscribe.
    verb = frame[:1]
    if verb == _SUBSCRIBE_BYTE:
      delta = 1
    elif verb == _UNSUBSCRIBE_BYTE:
      delta = -1
    else:
      return
    with self._presence_lock:
      self._presence[name] = max(0, self._presence[name] + delta)
