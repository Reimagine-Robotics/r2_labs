"""Regression tests for rpc_api shapes that ride the pickle wire."""

import pickle

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
