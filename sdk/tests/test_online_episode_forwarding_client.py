"""Tests for the online episode forwarding SDK client."""

import pickle

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


class _RecordingRpcClient:

  def __init__(self) -> None:
    self.calls: list[str] = []
    self.start_queries: list[rpc_api.OnlineEpisodeForwardingStartQuery] = []

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    del timeout
    self.calls.append(fn_name)
    if fn_name.endswith(".start"):
      query = pickle.loads(data or b"")
      assert isinstance(query, rpc_api.OnlineEpisodeForwardingStartQuery)
      self.start_queries.append(query)
    return pickle.dumps(
        rpc_api.OnlineEpisodeForwardingStateResponse(
            server_address="tcp://trainer",
            enabled=fn_name.endswith(".start"),
            generation="online_learning_2",
        )
    )


def test_lifecycle_calls_the_forwarding_endpoints() -> None:
  rpc_client = _RecordingRpcClient()
  forwarding = sdk_client.OnlineEpisodeForwardingClient(
      rpc_client  # type: ignore[arg-type]
  )

  started = forwarding.start("skill_online_learning_2#model")
  stopped = forwarding.stop()
  forwarding.get_state()

  assert started.enabled
  assert not stopped.enabled
  assert rpc_client.calls == [
      "online_episode_forwarding.start",
      "online_episode_forwarding.stop",
      "online_episode_forwarding.get_state",
  ]
  assert rpc_client.start_queries == [
      rpc_api.OnlineEpisodeForwardingStartQuery(
          model_name="skill_online_learning_2#model"
      )
  ]
