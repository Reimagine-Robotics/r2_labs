"""RPC server for handling remote function calls from robot clients."""

import dataclasses
import pickle
import time
from typing import Callable

import zmq
import zstd
from loguru import logger as log

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


@dataclasses.dataclass
class RpcArgs:
  """Arguments for an RPC call.

  Attributes:
    fn_name: Name of the remote function to invoke.
    fn_args: Serialized function arguments, or None for no-arg calls.
    use_compression: Whether the payload is zstd compressed.
  """

  fn_name: str
  fn_args: bytes | None
  use_compression: bool = False


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

    self._fn_registry = {}

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

  def run(self) -> None:
    """Run the server message loop. Blocks indefinitely."""
    while True:
      message = self._socket.recv()
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

      self._socket.send(result)

  def _ping(self) -> bytes:
    """Handle ping requests for server health checks."""
    return pickle.dumps("ack")
