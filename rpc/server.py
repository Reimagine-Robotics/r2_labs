"""RPC server for handling remote function calls from robot clients."""

import dataclasses
import pickle
import struct
import time
from typing import Callable

import prometheus_client
import zmq
import zstd
from loguru import logger as log

from r2_labs import version

# We need to find a balance of ZSTD compression between taking longer to do the
# compression vs a smaller payload over the network. Depending on the specific
# network speed and client/server CPU speeds the optimal value for this may
# differ. Higher values require more compute but result in smaller payload.
# Valid values are -7 .. 22. Library default is around 3.
ZSTD_COMPRESSION_LEVEL = -1

# Handlers slower than this block the single-threaded REQ/REP loop and risk
# causing peer-handler timeouts on the client. Log a warning when it happens
# so the responsible handler is identifiable from server logs.
SLOW_HANDLER_THRESHOLD_S = 1.0

# How long run() waits per poll for a request before re-checking stop().
_STOP_POLL_INTERVAL_MS = 100

# rpc metric group, server side. Recorded unconditionally (in-memory only);
# whether anything scrapes them is deployment configuration.
_HANDLER_DURATION = prometheus_client.Histogram(
    "r2_rpc_handler_duration_seconds",
    "Wall time spent inside one RPC handler function.",
    ["handler"],
)
_SERVER_REQUESTS = prometheus_client.Counter(
    "r2_rpc_server_requests_total",
    "RPC requests received, by handler.",
    ["handler"],
)
_SERVER_BACKLOGGED_REQUESTS = prometheus_client.Counter(
    "r2_rpc_server_backlogged_requests_total",
    "RPC requests already waiting in the socket buffer when the server"
    " finished its previous handler, i.e. requests that queued behind a busy"
    " server rather than arriving while it was idle.",
    ["handler"],
)


@dataclasses.dataclass
class RpcArgs:
  """Arguments for an RPC call.

  Attributes:
    fn_name: Name of the remote function to invoke.
    fn_args: Serialized function arguments, or None for no-arg calls.
    use_compression: Whether the payload is zstd compressed.
    return_busy_time: Ask the server to append its busy time (seconds from
      request receipt to reply-ready) to the reply as an extra message frame.
      Old servers unpickle this field into an unused attribute and reply with
      a single frame, so the protocol degrades rather than breaks.
  """

  fn_name: str
  fn_args: bytes | None
  use_compression: bool = False
  return_busy_time: bool = False


@dataclasses.dataclass
class RpcError:
  """Error response from RPC server when a handler raises an exception.

  Attributes:
    message: The exception message from the server.
  """

  message: str


class BaseServer:
  """ZMQ-based RPC server using REQ/REP pattern.

  Register handler functions with `register_fn`, then call `run` to start
  the message loop. A "ping" endpoint is registered automatically.
  """

  def __init__(self, port: int, context: zmq.Context | None):
    if context is None:
      context = zmq.Context()

    self._socket = context.socket(zmq.REP)
    self._socket.bind(f"tcp://*:{port}")
    # Actual bound port. Equals `port`, or the OS-assigned one when port=0 is
    # passed to bind an ephemeral port (lets callers/tests bind-and-discover
    # without a racy pick-a-free-port-then-rebind dance).
    endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
    self.port = int(endpoint.rsplit(":", 1)[1])

    self._fn_registry = {}
    self._stop_requested = False

    # Always register a "ping" end-point with every server. This allows the
    # client to easily check whether the server is up.
    self.register_fn(fn=self._ping, fn_name="ping")

  def register_fn(
      self,
      fn: Callable[[], bytes] | Callable[[bytes], bytes],
      fn_name: str | None = None,
  ) -> None:
    """Register a function as an RPC endpoint.

    Args:
      fn: Handler function that takes optional bytes and returns bytes.
      fn_name: Endpoint name. Defaults to the function's __name__.
    """
    if fn_name is None:
      fn_name = fn.__name__

    # log.info("Registered RPC function: {}", fn_name)
    self._fn_registry[fn_name] = fn

  def stop(self) -> None:
    """Ask run() to return after the in-flight request, if any.

    Lets an owning thread shut the loop down and join it before tearing the
    ZMQ context down; destroying the context under a parked recv aborts the
    process inside libzmq rather than raising.
    """
    self._stop_requested = True

  def run(self) -> None:
    """Run the server message loop. Blocks until stop() is called."""
    while not self._stop_requested:
      # A request already sitting in the buffer at this zero-timeout poll
      # arrived while the server was busy (or starting up), so it queued
      # behind another handler; one that arrives during the idle wait below
      # found the server free and never queued.
      backlogged = self._socket.poll(0) != 0
      if not backlogged:
        while (
            not self._stop_requested
            and self._socket.poll(_STOP_POLL_INTERVAL_MS) == 0
        ):
          pass
        if self._stop_requested:
          break
      message = self._socket.recv()
      # Busy time spans request receipt to reply-ready, so decompression,
      # dispatch, and reply compression all count as server time rather than
      # leaking into the client's overhead measurement.
      t_busy_start = time.perf_counter()
      args = pickle.loads(message)
      assert isinstance(args, RpcArgs)

      fn_args = args.fn_args
      if args.use_compression and fn_args is not None:
        fn_args = zstd.decompress(fn_args)

      t_start = time.perf_counter()
      try:
        if fn_args is None:
          result = self._fn_registry[args.fn_name]()
        else:
          result = self._fn_registry[args.fn_name](fn_args)
      except Exception as e:
        log.exception("RPC handler {} raised exception", args.fn_name)
        result = pickle.dumps(RpcError(str(e)))
      finally:
        elapsed = time.perf_counter() - t_start
        _HANDLER_DURATION.labels(handler=args.fn_name).observe(elapsed)
        _SERVER_REQUESTS.labels(handler=args.fn_name).inc()
        if backlogged:
          _SERVER_BACKLOGGED_REQUESTS.labels(handler=args.fn_name).inc()
        if elapsed > SLOW_HANDLER_THRESHOLD_S:
          log.warning(
              "RPC handler {} took {:.3f}s (>{:.1f}s threshold); blocks"
              " other handlers and may cause client-side timeouts.",
              args.fn_name,
              elapsed,
              SLOW_HANDLER_THRESHOLD_S,
          )

      if args.use_compression:
        result = zstd.compress(result, ZSTD_COMPRESSION_LEVEL)

      # getattr rather than attribute access: an old client's pickled RpcArgs
      # predates the field, and dataclass defaults don't apply on unpickle.
      if getattr(args, "return_busy_time", False):
        busy_seconds = time.perf_counter() - t_busy_start
        self._socket.send_multipart([result, struct.pack("<d", busy_seconds)])
      else:
        self._socket.send(result)

  def _ping(self) -> bytes:
    """Handle ping requests for server health checks.

    Returns the server's r2_labs version alongside the legacy "ack" status so
    clients can warn on a version mismatch. version is None when the server
    can't determine its own version. Kept a plain dict (not a dataclass) so old
    clients that only logged the reply still work.
    """
    return pickle.dumps({"status": "ack", "version": version.get_version()})
