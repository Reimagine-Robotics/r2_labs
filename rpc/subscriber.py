"""ZMQ subscribers for the pub/sub broadcast plane.

`BaseSubscriber` consumes a publisher socket synchronously (one per thread);
`AsyncBaseSubscriber` is its `zmq.asyncio` sibling for event-loop consumers
like the streamer. Wrap either in a typed `*EventStream` that unpickles
`payload` into a domain object.
"""

from typing import AsyncIterator, Iterator

import zmq
import zmq.asyncio


class BaseSubscriber:
  """Subscribes to a publisher socket and yields `(topic, payload)` pairs."""

  def __init__(self, address: str, context: zmq.Context | None = None):
    self._context = context if context is not None else zmq.Context()
    self._socket = self._context.socket(zmq.SUB)
    self._socket.setsockopt(zmq.LINGER, 0)
    self._socket.connect(address)

  def subscribe(self, topic: str) -> None:
    """Subscribe to a topic prefix.

    ZMQ pub/sub has a slow-joiner caveat: messages a publisher sends before
    this subscription propagates to it are not delivered — and no high-water
    mark prevents that, since the loss is from the absent subscription, not a
    full buffer. So a consumer that can't miss a *discrete event* must
    subscribe before that event can occur. For *evolving state*, which pub/sub
    never replays, subscribing isn't enough on its own: subscribe first, then
    fetch the current value once on connect.
    """
    self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())

  def unsubscribe(self, topic: str) -> None:
    self._socket.setsockopt(zmq.UNSUBSCRIBE, topic.encode())

  def recv(self, timeout_ms: int | None = None) -> tuple[str, bytes] | None:
    """Receive the next message, or None if `timeout_ms` elapses first."""
    if timeout_ms is not None and not self._socket.poll(timeout_ms):
      return None
    topic, payload = self._socket.recv_multipart()
    return topic.decode(), payload

  def __iter__(self) -> Iterator[tuple[str, bytes]]:
    while True:
      topic, payload = self._socket.recv_multipart()
      yield topic.decode(), payload

  def close(self) -> None:
    self._socket.close()


class AsyncBaseSubscriber:
  """Async sibling of `BaseSubscriber` for event-loop consumers (the streamer).

  Connects one `zmq.asyncio` SUB across one or more publisher sockets and
  `await`s messages, so a single event loop can consume the feed while also
  serving HTTP. `subscribe`/`unsubscribe` may be called dynamically (e.g.
  demand-driven as browsers connect). Same slow-joiner caveat as
  `BaseSubscriber.subscribe`.
  """

  def __init__(
      self,
      addresses: str | list[str],
      context: zmq.asyncio.Context | None = None,
  ):
    self._context = context if context is not None else zmq.asyncio.Context()
    self._socket = self._context.socket(zmq.SUB)
    self._socket.setsockopt(zmq.LINGER, 0)
    for address in [addresses] if isinstance(addresses, str) else addresses:
      self._socket.connect(address)

  def subscribe(self, topic: str) -> None:
    self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())

  def unsubscribe(self, topic: str) -> None:
    self._socket.setsockopt(zmq.UNSUBSCRIBE, topic.encode())

  async def recv(self) -> tuple[str, bytes]:
    topic, payload = await self._socket.recv_multipart()
    return topic.decode(), payload

  def __aiter__(self) -> AsyncIterator[tuple[str, bytes]]:
    return self

  async def __anext__(self) -> tuple[str, bytes]:
    return await self.recv()

  def close(self) -> None:
    self._socket.close()
