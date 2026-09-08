"""Tests for learned-skill trainer selection in the SDK client."""

import inspect
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
    if fn_name == "trainer.start_current_export":
      response = rpc_api.StartExportResponse()
    elif "status" in fn_name:
      response = rpc_api.TrainingStatusResponse(
          is_finished=False,
          steps_completed=0,
          max_steps=1,
          loss=0.0,
          fps=0.0,
          seconds_per_step=0.0,
      )
    elif "cancel" in fn_name:
      response = rpc_api.CancelTrainingResponse(success=True)
    else:
      response = rpc_api.StartSkillTrainingResponse()
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


def test_online_learning_uses_consistent_endpoint_and_request_names():
  trainer, rpc_client = _trainer_client()

  trainer.start_online_learning(
      "skill#model",
      inference_gpu=2,
      inference_port=4243,
      restart_online_learning=True,
      external_task_id="task",
  )

  endpoint, request, _ = rpc_client.calls[0]
  assert endpoint == "trainer.start_online_learning"
  assert isinstance(request, rpc_api.StartSkillTrainingQuery)
  assert request.restart_online_learning
  assert request.serve_online_learning_inference
  assert request.online_learning_inference_gpu == 2
  assert request.online_learning_inference_port == 4243
  assert request.external_task_id == "task"


def test_online_learning_lifecycle_uses_consistent_endpoints():
  trainer, rpc_client = _trainer_client()

  trainer.get_online_learning_status()
  trainer.cancel_online_learning()

  assert [call[0] for call in rpc_client.calls] == [
      "trainer.get_online_learning_status",
      "trainer.cancel_online_learning",
  ]


def test_online_training_preserves_the_existing_timeout_position():
  parameters = list(
      inspect.signature(
          sdk_client.TrainerClient.start_online_training
      ).parameters
  )

  assert parameters[-2:] == ["timeout", "restart_online_learning"]


def test_online_learning_fields_default_when_unpickling_older_payloads():
  query = rpc_api.StartSkillTrainingQuery.__new__(
      rpc_api.StartSkillTrainingQuery
  )
  query.__setstate__({"model_name": "skill", "training_steps": 1})
  response = rpc_api.StartSkillTrainingResponse.__new__(
      rpc_api.StartSkillTrainingResponse
  )
  response.__setstate__({"error": None})

  assert not query.restart_online_learning
  assert not query.serve_online_learning_inference
  assert query.online_learning_inference_gpu is None
  assert query.online_learning_inference_port is None
  assert response.online_learning_inference_address is None
