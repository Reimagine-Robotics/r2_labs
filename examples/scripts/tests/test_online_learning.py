"""Tests for selecting the robot's forwarding destination in the example."""

from unittest import mock

import pytest
from absl.testing import flagsaver

from r2_labs.examples.scripts import online_learning


@pytest.mark.parametrize("trainer_hostname", [None, "atlas.local"])
def test_attach_preserves_the_default_unless_a_trainer_is_explicit(
    monkeypatch: pytest.MonkeyPatch, trainer_hostname: str | None
) -> None:
  robot = mock.Mock()
  attach = mock.Mock()
  monkeypatch.setattr(online_learning, "connect", mock.Mock(return_value=robot))
  monkeypatch.setattr(online_learning, "attach_forwarding", attach)
  monkeypatch.setattr(
      online_learning,
      "follow_session",
      mock.Mock(return_value=mock.Mock(phase="complete")),
  )
  monkeypatch.setattr(online_learning, "detach_forwarding", mock.Mock())
  args = ["online_learning", "--mode=attach", "--robot_hostname=cypher.local"]
  if trainer_hostname is not None:
    args.append(f"--training_server_hostname={trainer_hostname}")

  with flagsaver.flagsaver():
    online_learning.FLAGS(args)
    online_learning.main([])

  attach.assert_called_once_with(
      robot,
      server_address=(
          f"tcp://{trainer_hostname}:7534"
          if trainer_hostname is not None
          else None
      ),
  )
