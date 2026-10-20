"""Tests for the online episode forwarding SDK client."""

import pickle
from unittest import mock

import pytest

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


class _RecordingRpcClient:

  server_version: str | None = None

  def __init__(self, server_address: str = "tcp://trainer") -> None:
    self.calls: list[str] = []
    self.start_queries: list[rpc_api.OnlineEpisodeForwardingStartQuery] = []
    self.server_address = server_address

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
            server_address=self.server_address,
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


def test_start_passes_the_requested_trainer_address() -> None:
  rpc_client = _RecordingRpcClient("tcp://atlas")
  forwarding = sdk_client.OnlineEpisodeForwardingClient(
      rpc_client  # type: ignore[arg-type]
  )

  state = forwarding.start(
      "skill_online_learning_2#model",
      "base#one",
      server_address="tcp://atlas",
  )

  assert state.error is None
  assert rpc_client.start_queries == [
      rpc_api.OnlineEpisodeForwardingStartQuery(
          model_name="skill_online_learning_2#model",
          base_model_id="base#one",
          server_address="tcp://atlas",
      )
  ]


@pytest.mark.parametrize("server_address", [None, "tcp://atlas"])
@pytest.mark.parametrize("model_name", ["session", "replaced_session"])
def test_attach_captures_base_identity_on_client(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    server_address: str | None,
) -> None:
  rpc_client = _RecordingRpcClient(server_address or "tcp://trainer")
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
  create_client = mock.Mock(return_value=training_rpc)
  monkeypatch.setattr(sdk_client.client, "BaseClient", create_client)
  default_training_client = mock.Mock(return_value=training_rpc)
  forwarding = sdk_client.OnlineEpisodeForwardingClient(
      rpc_client,  # type: ignore[arg-type]
      default_training_client,
  )
  if model_name != "session":
    with pytest.raises(ValueError, match="session changed"):
      forwarding.start("session", server_address=server_address)
    assert not rpc_client.start_queries
  else:
    state = forwarding.start("session", server_address=server_address)
    assert state.error is None
    assert rpc_client.start_queries[0].base_model_id == "base#one"
    assert rpc_client.start_queries[0].server_address == server_address
  if server_address is None:
    default_training_client.assert_called_once_with()
    create_client.assert_not_called()
  else:
    create_client.assert_called_once_with(
        server_address, timeout=5000, ping_on_init=False
    )
    default_training_client.assert_not_called()
  training_rpc.reset_mock()
  forwarding.stop()
  training_rpc.assert_not_called()


def test_robot_preserves_timeout_when_selecting_a_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  robot_rpc = _RecordingRpcClient("tcp://atlas")
  training_rpc = mock.Mock(
      return_value=pickle.dumps(
          rpc_api.TrainingStatusResponse(
              is_finished=False,
              steps_completed=1,
              max_steps=10,
              loss=0,
              fps=1,
              seconds_per_step=1,
              model_name="session",
              base_model_id="base#one",
              online_mode=True,
          )
      )
  )
  create_client = mock.Mock(side_effect=[robot_rpc, training_rpc])
  monkeypatch.setattr(sdk_client.client, "BaseClient", create_client)
  robot = sdk_client.Robot(
      server_address="tcp://robot",
      query_server_address="tcp://query",
      training_server_address="tcp://trainer",
      timeout=30_000,
  )

  state = robot.online_episode_forwarding.start(
      "session", server_address="tcp://atlas"
  )

  assert state.enabled
  assert state.error is None
  assert create_client.call_args == mock.call(
      "tcp://atlas", timeout=30_000, ping_on_init=False
  )
