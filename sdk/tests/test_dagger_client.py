"""Tests for the public DAgger workflow client."""

import dataclasses
from unittest import mock

from r2_labs import rpc_api
from r2_labs.sdk import client as sdk_client


def test_dagger_state_exposes_episode_and_control_axes() -> None:
  field_names = {
      field.name for field in dataclasses.fields(rpc_api.DaggerStateResponse)
  }

  assert "episode_phase" in field_names
  assert "control_phase" in field_names
  assert "phase" not in field_names
  assert "leader_kind" not in field_names


def test_dagger_config_excludes_control_device_alignment() -> None:
  field_names = {
      field.name for field in dataclasses.fields(rpc_api.DaggerConfigQuery)
  }

  assert "align_timeout_seconds" not in field_names
  assert "align_threshold" not in field_names


def test_dagger_client_uses_workflow_rpc_interface(monkeypatch) -> None:
  calls: list[tuple[str, object | None]] = []

  def rpc_call(
      _client: object,
      fn_name: str,
      query: object | None = None,
      **_kwargs: object,
  ) -> object:
    calls.append((fn_name, query))
    if fn_name == "dagger.advance":
      return rpc_api.DaggerAdvanceResponse()
    if fn_name == "dagger.finish_episode":
      return rpc_api.DaggerFinishEpisodeResponse()
    if fn_name == "dagger.abort":
      return rpc_api.DaggerAbortResponse()
    raise AssertionError(fn_name)

  monkeypatch.setattr(sdk_client, "_rpc_call", rpc_call)
  dagger = sdk_client.DaggerClient(mock.Mock())
  finish_query = rpc_api.DaggerFinishEpisodeQuery(
      disposition=rpc_api.DaggerEpisodeDisposition.DISCARD
  )

  dagger.advance()
  dagger.finish_episode(finish_query)
  dagger.abort()

  assert calls == [
      ("dagger.advance", None),
      ("dagger.finish_episode", finish_query),
      ("dagger.abort", None),
  ]
  assert not hasattr(dagger, "toggle")
  assert not hasattr(dagger, "finalize")
  assert not hasattr(dagger, "stop")
