"""Tests for the online episode forwarding SDK client."""

import pickle
from unittest import mock

import pytest

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


@pytest.mark.parametrize("model_name", ["session", "replaced_session"])
def test_attach_captures_base_identity_on_client(model_name: str) -> None:
  rpc_client = _RecordingRpcClient()
  training_rpc = mock.Mock(
      return_value=pickle.dumps(
          rpc_api.TrainingStatusResponse(
              is_finished=False,
              steps_completed=1,
              max_steps=10,
              loss=0,
              fps=1,
              seconds_per_step=1,
              model_name=model_name,
              base_model_id="base#one",
              online_mode=True,
          )
      )
  )
  forwarding = sdk_client.OnlineEpisodeForwardingClient(
      rpc_client,  # type: ignore[arg-type]
      lambda: training_rpc,
  )
  if model_name != "session":
    with pytest.raises(ValueError, match="session changed"):
      forwarding.start("session")
    assert not rpc_client.start_queries
  else:
    forwarding.start("session")
    assert rpc_client.start_queries[0].base_model_id == "base#one"
  training_rpc.reset_mock()
  forwarding.stop()
  training_rpc.assert_not_called()
