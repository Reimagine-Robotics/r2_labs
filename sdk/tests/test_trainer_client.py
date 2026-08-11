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
    response = (
        rpc_api.StartExportResponse()
        if fn_name == "trainer.start_current_export"
        else rpc_api.StartSkillTrainingResponse()
    )
    return pickle.dumps(response)


def _trainer_client() -> tuple[sdk_client.TrainerClient, _RecordingRpcClient]:
  rpc_client = _RecordingRpcClient()
  trainer = sdk_client.TrainerClient(rpc_client)  # type: ignore[arg-type]
  trainer.is_training_running = lambda: False  # type: ignore[method-assign]
  return trainer, rpc_client


def test_demo_conditioned_uses_compatible_payload_by_default():
  trainer, rpc_client = _trainer_client()

  trainer.train_skill_model("skill", 10, ["episode*"])

  endpoint, request, _ = rpc_client.calls[0]
  assert endpoint == "trainer.train_skill_model"
  assert isinstance(request, rpc_api.StartSkillTrainingQuery)
  assert request.model_name == "skill"


def test_unconditioned_uses_shared_endpoint_and_preserves_overrides():
  trainer, rpc_client = _trainer_client()
  overrides = {"model.width": 128, "optimizer.learning_rate": 1e-5}

  trainer.train_skill_model(
      "skill",
      10,
      ["episode*"],
      model_family="unconditioned",
      config_overrides=overrides,
  )

  endpoint, request, _ = rpc_client.calls[0]
  assert endpoint == "trainer.train_skill_model"
  assert isinstance(request, rpc_api.StartOfflineSkillTrainingQuery)
  assert request.model_family == "unconditioned"
  assert request.training.config_overrides == overrides


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


def test_current_export_uses_explicit_current_export_query():
  trainer, rpc_client = _trainer_client()

  trainer.start_current_export(checkpoint_step=20)

  endpoint, query, _ = rpc_client.calls[0]
  assert endpoint == "trainer.start_current_export"
  assert query == rpc_api.StartCurrentExportQuery(checkpoint_step=20)
