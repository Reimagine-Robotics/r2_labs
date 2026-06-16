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
