import pickle
from unittest import mock

import pytest

import r2_labs
from r2_labs.sdk import client, rpc_api


def test_wait_for_ticket_raises_typed_force_failure() -> None:
  message = "applied force 63.0 N exceeded the limit of 52.0 N"
  peak = 63.0
  status = rpc_api.TicketStatusResponse(
      info=rpc_api.TicketInfo(
          ticket_id="force",
          status=rpc_api.TicketStatus.FAILED,
          behaviour_type="visual_trajectory_motion",
          created_at=0.0,
          termination_reason="FAILURE",
          error_message=message,
          result_data={
              "max_applied_force": 52.0,
              "peak_applied_force": peak,
          },
      )
  )
  rpc = mock.Mock(return_value=pickle.dumps(status))
  behaviour = client.BehaviourClient(lambda: rpc)
  with pytest.raises(r2_labs.AppliedForceLimitError) as caught:
    behaviour.wait_for_ticket("force")
  error = caught.value
  assert isinstance(error, client.BehaviourFailedError)
  assert error.peak_applied_force == peak
  assert error.max_applied_force == 52.0
  assert error.ticket_id == "force"
  assert error.termination_reason == "FAILURE"
  assert message in str(error)
