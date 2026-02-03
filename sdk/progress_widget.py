"""Progress display widgets for training monitoring.

Provides clean, reusable widgets for displaying training progress in Jupyter
notebooks. Follows separation of concerns - this module only handles display,
not data fetching.

Usage:
    from r2_labs.sdk.progress_widget import TrainingProgressWidget

    widget = TrainingProgressWidget(trainer_client)
    widget.start()  # Starts polling and displays progress
    # ... training runs ...
    widget.stop()   # Stop polling when done
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from r2_labs.sdk import rpc_api


class TrainingStatusProvider(Protocol):
  """Protocol for objects that provide training status."""

  def get_training_status(self) -> rpc_api.ProgressTrainingStatusResponse:
    """Get current training status."""
    ...


class TrainingProgressWidget:
  """Interactive widget for monitoring training progress in Jupyter notebooks.

  Polls training status and displays progress with a clean, minimal UI.
  Uses ipywidgets for interactive display.

  Example:
      trainer = ProgressPredictionTrainerClient(rpc_client)
      trainer.train_model(...)

      widget = TrainingProgressWidget(trainer)
      widget.start()  # Shows live progress
      # Training completes or widget.stop() called
  """

  def __init__(
      self,
      status_provider: TrainingStatusProvider,
      poll_interval: float = 2.0,
  ) -> None:
    """Initialize the progress widget.

    Args:
        status_provider: Object with get_training_status() method.
        poll_interval: Seconds between status polls.
    """
    self._provider = status_provider
    self._poll_interval = poll_interval
    self._stop_event = threading.Event()
    self._poll_thread: threading.Thread | None = None
    self._widgets_available = False
    self._widgets: dict | None = None

    # Try to import ipywidgets
    try:
      import ipywidgets as widgets
      from IPython.display import display

      self._widgets_module = widgets
      self._display = display
      self._widgets_available = True
    except ImportError:
      pass

  def _create_widgets(self) -> dict:
    """Create the ipywidgets UI components."""
    widgets = self._widgets_module

    # Phase indicator
    phase_label = widgets.HTML(
        value="<b>Phase:</b> Idle",
        layout=widgets.Layout(margin="0 0 5px 0"),
    )

    # Progress bar
    progress_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description="Progress:",
        bar_style="info",
        style={"bar_color": "#3498db"},
        layout=widgets.Layout(width="100%"),
    )

    # Step counter
    step_label = widgets.HTML(value="<b>Steps:</b> 0 / 0")

    # Metrics display
    metrics_label = widgets.HTML(
        value="<b>Loss:</b> - | <b>Acc:</b> - | <b>F1:</b> -"
    )

    # Speed and ETA
    speed_label = widgets.HTML(
        value="<b>Speed:</b> - steps/sec | <b>ETA:</b> -"
    )

    # Error display (hidden by default)
    error_label = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none"),
    )

    # Container
    container = widgets.VBox(
        [
            phase_label,
            progress_bar,
            step_label,
            metrics_label,
            speed_label,
            error_label,
        ],
        layout=widgets.Layout(
            padding="10px",
            border="1px solid #ddd",
            border_radius="5px",
            width="500px",
        ),
    )

    return {
        "container": container,
        "phase": phase_label,
        "progress": progress_bar,
        "steps": step_label,
        "metrics": metrics_label,
        "speed": speed_label,
        "error": error_label,
    }

  def _format_eta(self, seconds: float) -> str:
    """Format ETA in human-readable form."""
    if seconds < 60:
      return f"{seconds:.0f}s"
    elif seconds < 3600:
      return f"{seconds / 60:.1f}m"
    else:
      return f"{seconds / 3600:.1f}h"

  def _update_display(
      self, status: rpc_api.ProgressTrainingStatusResponse
  ) -> None:
    """Update widget display with current status."""
    if not self._widgets:
      return

    w = self._widgets

    # Update phase with color coding
    phase = status.phase
    phase_colors = {
        "idle": "#95a5a6",
        "preparing_dataset": "#f39c12",
        "training": "#3498db",
        "finished": "#27ae60",
        "failed": "#e74c3c",
    }
    color = phase_colors.get(phase, "#95a5a6")
    phase_text = f'<b>Phase:</b> <span style="color:{color}">{phase}</span>'
    if status.checkpoint_id:
      phase_text += f" | <b>Checkpoint:</b> {status.checkpoint_id}"
    if phase == "finished" and status.exported_model_id is not None:
      phase_text += f" | <b>Model ID:</b> {status.exported_model_id}"
    w["phase"].value = phase_text

    # Update progress bar
    if status.max_steps > 0:
      progress_pct = (status.steps_completed / status.max_steps) * 100
      w["progress"].value = progress_pct

      # Change bar color based on phase
      if phase == "finished":
        w["progress"].bar_style = "success"
      elif phase == "failed":
        w["progress"].bar_style = "danger"
      elif phase == "preparing_dataset":
        w["progress"].bar_style = "warning"
      else:
        w["progress"].bar_style = "info"
    else:
      w["progress"].value = 0

    # Update step counter
    w["steps"].value = (
        f"<b>Steps:</b> {status.steps_completed:,} / {status.max_steps:,}"
    )

    # Update metrics (train)
    loss_str = f"{status.loss:.4f}" if status.loss else "-"
    acc_str = f"{status.accuracy:.2%}" if status.accuracy is not None else "-"
    f1_str = f"{status.f1:.2%}" if status.f1 is not None else "-"
    train_metrics = (
        f"<b>Train:</b> Loss {loss_str} | Acc {acc_str} | F1 {f1_str}"
    )

    # Update metrics (val)
    val_loss_str = (
        f"{status.val_loss:.4f}" if status.val_loss is not None else "-"
    )
    val_acc_str = (
        f"{status.val_accuracy:.2%}" if status.val_accuracy is not None else "-"
    )
    val_f1_str = f"{status.val_f1:.2%}" if status.val_f1 is not None else "-"
    val_metrics = (
        f"<b>Val:</b> Loss {val_loss_str} | Acc {val_acc_str} | F1 {val_f1_str}"
    )

    w["metrics"].value = f"{train_metrics}<br>{val_metrics}"

    # Update speed and ETA
    if status.fps and status.fps > 0:
      remaining_steps = status.max_steps - status.steps_completed
      eta_seconds = remaining_steps / status.fps if remaining_steps > 0 else 0
      eta_str = self._format_eta(eta_seconds) if eta_seconds > 0 else "-"
      w["speed"].value = (
          f"<b>Speed:</b> {status.fps:.1f} steps/sec | <b>ETA:</b> {eta_str}"
      )
    else:
      w["speed"].value = "<b>Speed:</b> - | <b>ETA:</b> -"

    # Update error display
    if status.error:
      w["error"].value = (
          f'<span style="color:#e74c3c"><b>Error:</b> {status.error}</span>'
      )
      w["error"].layout.display = "block"
    else:
      w["error"].layout.display = "none"

  def _print_status(
      self, status: rpc_api.ProgressTrainingStatusResponse
  ) -> None:
    """Print status to console (fallback when widgets unavailable)."""
    phase = status.phase
    progress_pct = (
        (status.steps_completed / status.max_steps * 100)
        if status.max_steps > 0
        else 0
    )

    # Build progress bar
    bar_width = 30
    filled = int(bar_width * progress_pct / 100)
    bar = "=" * filled + "-" * (bar_width - filled)

    # Build train metrics string
    train_metrics = []
    if status.loss:
      train_metrics.append(f"loss={status.loss:.4f}")
    if status.accuracy is not None:
      train_metrics.append(f"acc={status.accuracy:.2%}")
    if status.f1 is not None:
      train_metrics.append(f"f1={status.f1:.2%}")
    train_str = ", ".join(train_metrics) if train_metrics else "-"

    # Build val metrics string
    val_metrics = []
    if status.val_loss is not None:
      val_metrics.append(f"val_loss={status.val_loss:.4f}")
    if status.val_accuracy is not None:
      val_metrics.append(f"val_acc={status.val_accuracy:.2%}")
    if status.val_f1 is not None:
      val_metrics.append(f"val_f1={status.val_f1:.2%}")
    val_str = f" | {', '.join(val_metrics)}" if val_metrics else ""

    checkpoint_str = (
        f" | ckpt={status.checkpoint_id}" if status.checkpoint_id else ""
    )
    model_str = ""
    if phase == "finished" and status.exported_model_id is not None:
      model_str = f" | model_id={status.exported_model_id}"

    print(
        f"[{phase}] [{bar}] {progress_pct:.1f}% "
        f"({status.steps_completed}/{status.max_steps}) {train_str}{val_str}{checkpoint_str}{model_str}"
    )

    if status.error:
      print(f"ERROR: {status.error}")

  def _poll_loop(self) -> None:
    """Background thread that polls status and updates display."""
    while not self._stop_event.is_set():
      try:
        status = self._provider.get_training_status()

        if self._widgets_available:
          self._update_display(status)
        else:
          self._print_status(status)

        # Stop if finished or failed
        if status.is_finished or status.phase in ("finished", "failed"):
          break

      except Exception as e:
        if self._widgets_available and self._widgets:
          self._widgets["error"].value = (
              f'<span style="color:#e74c3c"><b>Poll error:</b> {e}</span>'
          )
          self._widgets["error"].layout.display = "block"
        else:
          print(f"Poll error: {e}")

      self._stop_event.wait(self._poll_interval)

  def start(self) -> None:
    """Start displaying progress.

    Creates widgets (if in notebook) and begins polling in background.
    """
    if self._poll_thread is not None and self._poll_thread.is_alive():
      return  # Already running

    self._stop_event.clear()

    if self._widgets_available:
      self._widgets = self._create_widgets()
      self._display(self._widgets["container"])

    self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
    self._poll_thread.start()

  def stop(self) -> None:
    """Stop polling and updating display."""
    self._stop_event.set()
    if self._poll_thread is not None:
      self._poll_thread.join(timeout=5.0)
      self._poll_thread = None

  def wait(self, timeout: float | None = None) -> bool:
    """Wait for training to complete.

    Args:
        timeout: Maximum seconds to wait, or None for unlimited.

    Returns:
        True if training finished, False if timeout or stopped.
    """
    if self._poll_thread is None:
      return True

    self._poll_thread.join(timeout=timeout)
    return not self._poll_thread.is_alive()


def monitor_training(
    status_provider: TrainingStatusProvider,
    poll_interval: float = 2.0,
    timeout: float | None = None,
) -> rpc_api.ProgressTrainingStatusResponse:
  """Convenience function to monitor training until completion.

  Creates a widget, displays it, and waits for training to finish.

  Args:
      status_provider: Object with get_training_status() method.
      poll_interval: Seconds between status polls.
      timeout: Maximum seconds to wait, or None for unlimited.

  Returns:
      Final training status.

  Example:
      trainer.train_model(...)
      final_status = monitor_training(trainer)
      print(f"Final accuracy: {final_status.accuracy}")
  """
  widget = TrainingProgressWidget(status_provider, poll_interval)
  widget.start()
  widget.wait(timeout)
  widget.stop()
  return status_provider.get_training_status()
