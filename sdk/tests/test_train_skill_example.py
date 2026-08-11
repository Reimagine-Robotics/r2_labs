"""Tests for the offline skill-training example's command-line interface."""

import pathlib
import subprocess
import sys


def test_offline_training_example_help() -> None:
  script = (
      pathlib.Path(__file__).parents[2]
      / "examples"
      / "scripts"
      / "train_skill.py"
  )

  result = subprocess.run(
      [sys.executable, str(script), "--help"],
      check=False,
      capture_output=True,
      text=True,
  )

  assert "--hostname" in result.stdout
  assert "--model_name" in result.stdout
  assert "--entry_filters" in result.stdout
  assert "--cameras" in result.stdout
  assert "--training_steps" in result.stdout
