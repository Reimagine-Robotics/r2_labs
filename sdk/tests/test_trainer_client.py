"""Tests for learned-skill trainer selection in the SDK client."""

import pickle

import pytest

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


class _RecordingRpcClient:

  def __init__(self):
    self.calls: list[tuple[str, object | None, int | None]] = []

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    query = pickle.loads(data) if data is not None else None
    self.calls.append((fn_name, query, timeout))
    return pickle.dumps(rpc_api.StartSkillTrainingResponse())


def _trainer_client() -> tuple[sdk_client.TrainerClient, _RecordingRpcClient]:
  rpc_client = _RecordingRpcClient()
  trainer = sdk_client.TrainerClient(rpc_client)  # type: ignore[arg-type]
  trainer.is_training_running = lambda: False  # type: ignore[method-assign]
  return trainer, rpc_client


def test_demo_conditioned_remains_the_default_endpoint():
  trainer, rpc_client = _trainer_client()

  trainer.train_skill_model("skill", 10, ["episode*"])

  assert rpc_client.calls[0][0] == "trainer.train_skill_model"


def test_unconditioned_uses_additive_endpoint_and_preserves_overrides():
  trainer, rpc_client = _trainer_client()
  overrides = {"model.width": 128, "optimizer.learning_rate": 1e-5}

  trainer.train_skill_model(
      "skill",
      10,
      ["episode*"],
      model_family="unconditioned",
      config_overrides=overrides,
  )

  endpoint, query, _ = rpc_client.calls[0]
  assert endpoint == "trainer.train_uncond_skill_model"
  assert isinstance(query, rpc_api.StartSkillTrainingQuery)
  assert query.config_overrides == overrides


def test_unknown_model_family_is_rejected_before_rpc():
  trainer, rpc_client = _trainer_client()

  with pytest.raises(ValueError, match="unknown skill model family"):
    trainer.train_skill_model(
        "skill",
        10,
        ["episode*"],
        model_family="other",  # type: ignore[arg-type]
    )

  assert rpc_client.calls == []
