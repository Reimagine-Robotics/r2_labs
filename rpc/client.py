"""RPC client for communicating with robot servers."""

import pickle

import zmq
import zstd
from loguru import logger as log

from r2_labs.rpc import server


class RpcRemoteError(Exception):
  """Raised when the RPC server handler raised an exception."""


class RpcTimeoutError(TimeoutError):
  """Raised when an RPC call times out."""


class BaseClient:
  """ZMQ-based RPC client for robot communication.

  Uses REQ/REP pattern with optional zstd compression. Automatically handles
  timeouts and socket recovery.
  """

  def __init__(
      self,
      server_address: str,
      timeout: int = 5000,
      use_compression: bool = False,
  ):
    """Initialize RPC client.

    Args:
      server_address: Server address (e.g., "tcp://localhost:4243")
      timeout: Timeout in milliseconds for recv operations. -1 means no timeout.
      use_compression: Whether to compress RPC payloads with zstd.
    """
    self._server_address = server_address
    self._timeout = timeout
    self._use_compression = use_compression
    self._context = zmq.Context()
    self._socket: zmq.Socket = None  # type: ignore[assignment]

    self._create_socket()
    self.ping_server()

  def _create_socket(self) -> None:
    """Create and configure a new REQ socket."""
    self._socket = self._context.socket(zmq.REQ)
    self._socket.connect(self._server_address)
    if self._timeout > 0:
      self._socket.setsockopt(zmq.SNDTIMEO, self._timeout)
      self._socket.setsockopt(zmq.RCVTIMEO, self._timeout)

  def _reset_socket(self) -> None:
    """Reset socket after a timeout to allow new requests."""
    if self._socket is not None:
      self._socket.setsockopt(zmq.LINGER, 0)
      self._socket.close()
    self._create_socket()

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
    if self._use_compression and data is not None:
      data = zstd.compress(data, server.ZSTD_COMPRESSION_LEVEL)

    rpc_args = server.RpcArgs(
        fn_name=fn_name,
        fn_args=data,
        use_compression=self._use_compression,
    )
    message = pickle.dumps(rpc_args)

    try:
      # apply per-call timeout if specified
      if timeout is not None:
        self._socket.setsockopt(zmq.SNDTIMEO, timeout)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout)

      self._socket.send(message)
      result = self._socket.recv()
    except zmq.Again as exc:
      log.warning("RPC timeout, resetting socket to {}", self._server_address)
      self._reset_socket()
      raise RpcTimeoutError(
          f"RPC timeout calling {fn_name} on {self._server_address}"
      ) from exc
    finally:
      # restore default timeout
      if timeout is not None and self._timeout > 0:
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout)

    if self._use_compression:
      result = zstd.decompress(result)

    # check if server returned an error
    try:
      maybe_error = pickle.loads(result)
      if isinstance(maybe_error, server.RpcError):
        raise RpcRemoteError(maybe_error.message)
    except pickle.UnpicklingError:
      pass  # not an error, return raw bytes

    return result

  def ping_server(self):
    """Pings the server to make sure it's up."""
    try:
      log.info("Pinging {}", self._server_address)
      reply = self(fn_name="ping")
      log.info("Sever reply: {}", pickle.loads(reply))
    except RpcTimeoutError as exc:
      log.warning("Server {} not responding.", self._server_address)
      raise exc
