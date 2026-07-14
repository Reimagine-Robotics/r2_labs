"""RPC client for communicating with robot servers."""

import dataclasses
import os
import pickle
import struct
import threading
import time

import prometheus_client
import zmq
import zstd
from loguru import logger as log

from r2_labs.rpc import server

_PROFILE: bool = os.environ.get("R2_PROFILE_INFERENCE", "") == "1"

# rpc metric group, client side. `service` is the constructor's service_name,
# falling back to the target host:port (both bounded sets).
_CLIENT_REQUEST_DURATION = prometheus_client.Histogram(
    "r2_rpc_client_request_duration_seconds",
    "Client wall time around one RPC call, pickling and compression included."
    " Failed calls are observed too (a timeout at the deadline, a remote"
    " error at the full exchange time), so an outage shows up in latency"
    " rather than only in the error counter.",
    ["service", "fn"],
)
_CLIENT_OVERHEAD = prometheus_client.Histogram(
    "r2_rpc_client_overhead_seconds",
    "Client send-to-reply wall time minus the server's busy time for the same"
    " call: network transit plus time queued behind other handlers. Absent"
    " against servers that predate the busy-time reply frame.",
    ["service", "fn"],
)
_CLIENT_ERRORS = prometheus_client.Counter(
    "r2_rpc_client_errors_total",
    "RPC calls that failed, split into local timeouts and remote handler"
    " exceptions.",
    ["service", "fn", "error"],
)


@dataclasses.dataclass
class RpcTimings:
  """Timings collected inside BaseClient.__call__."""

  t_compress: float = 0.0
  t_wrap: float = 0.0
  request_wire_bytes: int = 0
  t_send: float = 0.0
  t_recv: float = 0.0
  t_decompress: float = 0.0
  response_wire_bytes: int = 0


class RpcRemoteError(Exception):
  """Raised when the RPC server handler raised an exception."""


class RpcTimeoutError(TimeoutError):
  """Raised when an RPC call times out."""


class BaseClient:
  """ZMQ-based RPC client for robot communication.

  Uses REQ/REP pattern with optional zstd compression. Each calling thread
  gets its own ZMQ socket (via ``threading.local``) so concurrent callers
  from different threads don't block each other. This follows the ZMQ rule:
  "Do not use or close sockets except in the thread that created them."

  See: https://zguide.zeromq.org/docs/chapter2/#Multithreading-with-ZeroMQ
  """

  def __init__(
      self,
      server_address: str,
      timeout: int = 5000,
      use_compression: bool = False,
      service_name: str | None = None,
  ):
    """Initialize RPC client.

    Args:
      server_address: Server address (e.g., "tcp://localhost:4243")
      timeout: Timeout in milliseconds for recv operations. -1 means no timeout.
      use_compression: Whether to compress RPC payloads with zstd.
      service_name: Optional human-readable name for the remote service.
    """
    self._server_address = server_address
    self._timeout = timeout
    self._use_compression = use_compression
    self._service_name = service_name
    # Metric label for this client's series: the caller-supplied name, or the
    # target endpoint ("host:port") when no name was given.
    self._metrics_service = service_name or server_address.rsplit("/", 1)[-1]
    # ZMQ contexts are thread-safe and should be shared across threads.
    self._context = zmq.Context()
    self._local = threading.local()
    self._last_rpc_timings: RpcTimings | None = None
    # r2_labs version reported by the server's ping reply, or None for an older
    # server whose ping predates the version field. Read by the SDK to warn on
    # a version mismatch.
    self.server_version: str | None = None

    self.ping_server()

  def _create_socket(self) -> zmq.Socket:
    """Create and configure a new REQ socket owned by the calling thread."""
    sock = self._context.socket(zmq.REQ)
    sock.connect(self._server_address)
    if self._timeout > 0:
      sock.setsockopt(zmq.SNDTIMEO, self._timeout)
      sock.setsockopt(zmq.RCVTIMEO, self._timeout)
    return sock

  def _get_socket(self) -> zmq.Socket:
    """Get the calling thread's socket, creating one if needed."""
    sock = getattr(self._local, "socket", None)
    if sock is None:
      sock = self._create_socket()
      self._local.socket = sock
    return sock

  def _reset_socket(self) -> None:
    """Reset the calling thread's socket after a ZMQ error."""
    sock = getattr(self._local, "socket", None)
    if sock is not None:
      sock.setsockopt(zmq.LINGER, 0)
      sock.close()
    self._local.socket = self._create_socket()

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    """Make an RPC call.

    Args:
      fn_name: Name of the remote function to call.
      data: Serialized arguments to pass to the function.
      timeout: Override timeout in ms for this call. None uses default.
    """
    profile = _PROFILE
    t_call_start = time.perf_counter()

    if self._use_compression and data is not None:
      if profile:
        t0 = time.perf_counter()
      data = zstd.compress(data, server.ZSTD_COMPRESSION_LEVEL)
      if profile:
        t_compress = time.perf_counter() - t0  # type: ignore[possibly-unbound]
    else:
      t_compress = 0.0

    rpc_args = server.RpcArgs(
        fn_name=fn_name,
        fn_args=data,
        use_compression=self._use_compression,
        return_busy_time=True,
    )
    if profile:
      t0 = time.perf_counter()
    message = pickle.dumps(rpc_args)
    if profile:
      t_wrap = time.perf_counter() - t0  # type: ignore[possibly-unbound]
    else:
      t_wrap = 0.0

    request_wire_bytes = len(message)

    # The server's busy time for this call, read from the extra reply frame a
    # new server appends when asked; None against an old single-frame server.
    server_busy_seconds: float | None = None

    sock = self._get_socket()
    try:
      # apply per-call timeout if specified
      if timeout is not None:
        sock.setsockopt(zmq.SNDTIMEO, timeout)
        sock.setsockopt(zmq.RCVTIMEO, timeout)

      if profile:
        t0 = time.perf_counter()
      t_exchange_start = time.perf_counter()
      sock.send(message)
      if profile:
        t_send = time.perf_counter() - t0  # type: ignore[possibly-unbound]
        t0 = time.perf_counter()
      result = sock.recv()
      # ZMQ delivers multipart messages atomically, so once the first frame
      # arrived the rest are already buffered. Drain any frames beyond the
      # busy-time one to keep the REQ send/recv alternation intact.
      if sock.getsockopt(zmq.RCVMORE):
        busy_frame = sock.recv()
        while sock.getsockopt(zmq.RCVMORE):
          sock.recv()
        if len(busy_frame) == struct.calcsize("<d"):
          server_busy_seconds = struct.unpack("<d", busy_frame)[0]
      t_exchange_end = time.perf_counter()
    except zmq.ZMQError as exc:
      # Reset socket on any ZMQ error, not just timeouts. The REQ/REP
      # pattern requires strict send/recv alternation; if recv fails after
      # a successful send, the socket is stuck in "recv" state and all
      # subsequent sends will fail with EFSM.
      service_suffix = (
          f" (service: {self._service_name})" if self._service_name else ""
      )
      log.warning(
          "RPC error ({}), resetting socket to {}{}",
          exc,
          self._server_address,
          service_suffix,
      )
      self._reset_socket()
      if isinstance(exc, zmq.Again):
        _CLIENT_ERRORS.labels(
            service=self._metrics_service, fn=fn_name, error="timeout"
        ).inc()
        _CLIENT_REQUEST_DURATION.labels(
            service=self._metrics_service, fn=fn_name
        ).observe(time.perf_counter() - t_call_start)
        raise RpcTimeoutError(
            f"RPC timeout calling {fn_name} on"
            f" {self._server_address}{service_suffix}"
        ) from exc
      raise
    finally:
      # Restore default timeout on the original socket. After _reset_socket
      # the replacement already has default timeouts, so skip the restore.
      if timeout is not None and self._timeout > 0:
        if getattr(self._local, "socket", None) is sock:
          sock.setsockopt(zmq.SNDTIMEO, self._timeout)
          sock.setsockopt(zmq.RCVTIMEO, self._timeout)

    if profile:
      t_recv = time.perf_counter() - t0  # type: ignore[possibly-unbound]
    else:
      t_send = 0.0
      t_recv = 0.0

    response_wire_bytes = len(result)

    if self._use_compression:
      if profile:
        t0 = time.perf_counter()
      result = zstd.decompress(result)
      if profile:
        t_decompress = time.perf_counter() - t0  # type: ignore[possibly-unbound]
    else:
      t_decompress = 0.0

    if profile:
      self._last_rpc_timings = RpcTimings(
          t_compress=t_compress,
          t_wrap=t_wrap,
          request_wire_bytes=request_wire_bytes,
          t_send=t_send,
          t_recv=t_recv,
          t_decompress=t_decompress,
          response_wire_bytes=response_wire_bytes,
      )

    # check if server returned an error
    try:
      maybe_error = pickle.loads(result)
      if isinstance(maybe_error, server.RpcError):
        log.warning(
            "RPC remote error | fn={} error={}", fn_name, maybe_error.message
        )
        _CLIENT_ERRORS.labels(
            service=self._metrics_service, fn=fn_name, error="remote"
        ).inc()
        _CLIENT_REQUEST_DURATION.labels(
            service=self._metrics_service, fn=fn_name
        ).observe(time.perf_counter() - t_call_start)
        raise RpcRemoteError(maybe_error.message)
    except pickle.UnpicklingError:
      pass  # not an error, return raw bytes

    _CLIENT_REQUEST_DURATION.labels(
        service=self._metrics_service, fn=fn_name
    ).observe(time.perf_counter() - t_call_start)
    if server_busy_seconds is not None:
      # Both terms are durations measured on their own host, so no cross-host
      # clock sync is involved: this is what the caller waited for beyond the
      # server working (network transit plus queueing behind other handlers).
      _CLIENT_OVERHEAD.labels(
          service=self._metrics_service, fn=fn_name
      ).observe(
          max(0.0, (t_exchange_end - t_exchange_start) - server_busy_seconds)
      )

    return result

  def ping_server(self):
    """Pings the server to make sure it's up."""
    service_suffix = (
        f" (service: {self._service_name})" if self._service_name else ""
    )
    try:
      log.info("Pinging {}{}", self._server_address, service_suffix)
      reply = pickle.loads(self(fn_name="ping"))
      log.info("Sever reply: {}", reply)
      # New servers reply with a dict carrying the version; older servers reply
      # with the bare "ack" string (no version).
      if isinstance(reply, dict):
        self.server_version = reply.get("version")
    except RpcTimeoutError as exc:
      log.warning(
          "Server {} not responding.{}", self._server_address, service_suffix
      )
      raise exc
