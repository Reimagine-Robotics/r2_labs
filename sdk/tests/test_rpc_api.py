"""Regression tests for rpc_api shapes that ride the pickle wire."""

import pickle

import pytest

from r2_labs.sdk import rpc_api


def test_unset_survives_pickle_round_trip():
  # Partial-update queries use `UNSET` as a sentinel meaning "field
  # absent from this patch". The server side decides which fields to
  # forward via `query.X is rpc_api.UNSET`. If the sentinel doesn't
  # round-trip through pickle as the same singleton, every default
  # field leaks through the filter — the original bare `object()` was
  # reconstructed as a fresh instance on the unpickling side.
  payload = pickle.dumps(rpc_api.UNSET)
  restored = pickle.loads(payload)
  assert restored is rpc_api.UNSET


def test_unset_in_query_round_trips_as_singleton():
  # The realistic shape of the bug: an UpdateVisualTrajectoryObjectQuery
  # built with only an explicit start_idx must arrive on the server
  # with every untouched field still passing `is rpc_api.UNSET`.
  query = rpc_api.UpdateVisualTrajectoryObjectQuery(
      name="t", object_id="o", start_idx=3
  )
  restored = pickle.loads(pickle.dumps(query))

  assert restored.start_idx == 3
  assert restored.masks is rpc_api.UNSET
  assert restored.end_idx is rpc_api.UNSET
  assert restored.reference_type is rpc_api.UNSET
  assert restored.apriltag_metadata is rpc_api.UNSET
  assert restored.disp_name is rpc_api.UNSET


def test_trajectory_motion_rejects_period_and_playback_speed_together():
  # period_seconds and playback_speed are two ways of expressing the same
  # thing (duration); supplying both is a caller bug.
  with pytest.raises(ValueError, match="mutually exclusive"):
    rpc_api.TrajectoryMotionQuery(
        trajectory_name="t", period_seconds=1.0, playback_speed=2.0
    )


def test_trajectory_motion_rejects_non_positive_playback_speed():
  with pytest.raises(ValueError, match="positive"):
    rpc_api.TrajectoryMotionQuery(trajectory_name="t", playback_speed=0.0)


def test_trajectory_motion_accepts_each_timing_knob_alone():
  by_speed = rpc_api.TrajectoryMotionQuery(
      trajectory_name="t", playback_speed=2.0
  )
  assert by_speed.playback_speed == 2.0
  assert by_speed.period_seconds is None

  by_period = rpc_api.TrajectoryMotionQuery(
      trajectory_name="t", period_seconds=3.0
  )
  assert by_period.period_seconds == 3.0
  assert by_period.playback_speed is None


def test_trajectory_motion_playback_speed_survives_pickle_round_trip():
  # The query is pickled onto the wire. Python's default pickle protocol
  # restores instances via __new__ + __setstate__, so __post_init__ does
  # NOT re-run on the server — validation is client-side only. This test
  # ensures the field values are faithfully preserved across the round-trip.
  query = rpc_api.TrajectoryMotionQuery(trajectory_name="t", playback_speed=2.0)
  restored = pickle.loads(pickle.dumps(query))

  assert restored.playback_speed == 2.0
  assert restored.period_seconds is None


def _legacy_execution_mode_response() -> rpc_api.ExecutionModeQueryResponse:
  """A response as a server predating `available_modes` puts it on the wire.

  Pickle restores instances via __new__ + __setstate__, never __init__, so the
  state dict holds exactly the fields the SENDING side's class declared.
  Building it that way here reproduces the cross-version payload without
  vendoring a copy of the old class.
  """
  response = object.__new__(rpc_api.ExecutionModeQueryResponse)
  response.__setstate__({"current_mode": rpc_api.ExecutionMode.READY})
  return response


def test_legacy_execution_mode_response_defaults_to_every_mode():
  # Field defaults don't run on unpickle, so without __setstate__ the attribute
  # would be missing outright and every reader (the REST adapter included)
  # would raise AttributeError. The documented fallback is the pre-gating
  # behaviour: all modes offered, a doomed teleop request failing server-side
  # as it did before.
  restored = _legacy_execution_mode_response()

  assert restored.available_modes == list(rpc_api.ExecutionMode)


def test_execution_mode_available_modes_survives_pickle_round_trip():
  # A current server's value must pass through untouched — the default must not
  # shadow what was actually sent.
  teach_only = [
      rpc_api.ExecutionMode.STOP,
      rpc_api.ExecutionMode.READY,
      rpc_api.ExecutionMode.TEACH,
  ]
  response = rpc_api.ExecutionModeQueryResponse(
      current_mode=rpc_api.ExecutionMode.READY, available_modes=teach_only
  )

  restored = pickle.loads(pickle.dumps(response))

  assert restored.available_modes == teach_only


def test_empty_available_modes_is_not_mistaken_for_a_legacy_payload():
  # The default is applied on ABSENCE, not falsiness. A truthiness check would
  # promote a genuinely empty set to "every mode available" — re-enabling
  # teleop on precisely the robot that supports none of it.
  response = rpc_api.ExecutionModeQueryResponse(
      current_mode=rpc_api.ExecutionMode.STOP, available_modes=[]
  )

  restored = pickle.loads(pickle.dumps(response))

  assert restored.available_modes == []
