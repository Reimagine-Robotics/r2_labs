import dataclasses
import pickle
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


@dataclasses.dataclass
class RpcArgs:
  fn_name: str
  fn_args: bytes | None
  use_compression: bool = False


@dataclasses.dataclass
class RpcError:
  message: str


class BaseServer:

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

    if fn_name is None:
      fn_name = fn.__name__

    log.info("Registered RPC function: {}", fn_name)
    self._fn_registry[fn_name] = fn

  def run(self) -> None:
    while True:
      message = self._socket.recv()
      args = pickle.loads(message)
      assert isinstance(args, RpcArgs)

      fn_args = args.fn_args
      if args.use_compression and fn_args is not None:
        fn_args = zstd.decompress(fn_args)

      try:
        if fn_args is None:
          result = self._fn_registry[args.fn_name]()
        else:
          result = self._fn_registry[args.fn_name](fn_args)
      except Exception as e:
        log.exception("RPC handler {} raised exception", args.fn_name)
        result = pickle.dumps(RpcError(str(e)))

      if args.use_compression:
        result = zstd.compress(result, ZSTD_COMPRESSION_LEVEL)

      self._socket.send(result)

  def _ping(self) -> bytes:
    return pickle.dumps("ack")
