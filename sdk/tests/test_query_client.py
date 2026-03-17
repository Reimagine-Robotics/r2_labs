"""Tests for QueryClient error propagation."""

import pickle

from r2_labs.sdk import client as sdk_client
from r2_labs.sdk import rpc_api


class _FakeRpcClient:
  """Fake RPC client that returns a pre-configured pickled response."""

  def __init__(self, response: object) -> None:
    self._response = pickle.dumps(response)

  def __call__(
      self,
      fn_name: str,
      data: bytes | None = None,
      timeout: int | None = None,
  ) -> bytes:
    return self._response


def test_can_see_object_success() -> None:
  response = rpc_api.CanSeeObjectResponse(visible=True, object_name="cup")
  query_client = sdk_client.QueryClient(_FakeRpcClient(response))  # type: ignore[arg-type]
  result = query_client.can_see_object(["cup"])
  assert result.visible is True
  assert result.object_name == "cup"
  assert result.error is None


def test_can_see_object_error() -> None:
  response = rpc_api.CanSeeObjectResponse(
      visible=False, error="timed out querying detector"
  )
  query_client = sdk_client.QueryClient(_FakeRpcClient(response))  # type: ignore[arg-type]
  result = query_client.can_see_object(["cup"])
  assert result.visible is False
  assert result.error == "timed out querying detector"


def test_predict_progress_success() -> None:
  response = rpc_api.PredictProgressResponse(progress=0.5)
  query_client = sdk_client.QueryClient(_FakeRpcClient(response))  # type: ignore[arg-type]
  result = query_client.predict_progress(model_id="test_model")
  assert result.progress == 0.5
  assert result.error is None


def test_predict_progress_error() -> None:
  response = rpc_api.PredictProgressResponse(
      progress=None, error="model not available"
  )
  query_client = sdk_client.QueryClient(_FakeRpcClient(response))  # type: ignore[arg-type]
  result = query_client.predict_progress(model_id="test_model")
  assert result.progress is None
  assert result.error == "model not available"
