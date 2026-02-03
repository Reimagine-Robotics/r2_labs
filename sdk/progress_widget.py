"""Progress display widgets for training monitoring.

Provides clean, reusable widgets for displaying training progress in Jupyter
notebooks. Follows separation of concerns - this module only handles display,
not data fetching.

Usage:
    from r2_labs.sdk.progress_widget import TrainingProgressWidget

    # For progress prediction training:
    widget = TrainingProgressWidget(progress_trainer_client)
    widget.start()  # Starts polling and displays progress
    # ... training runs ...
    widget.stop()   # Stop polling when done

    # For skill training:
    from r2_labs.sdk.progress_widget import SkillTrainingProgressWidget

    widget = SkillTrainingProgressWidget(trainer_client)
    widget.start()
    widget.wait()  # Wait for completion
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
  from r2_labs.sdk import rpc_api


class SkillTrainingStatusProvider(Protocol):
  """Protocol for objects that provide skill training status."""

  def get_training_status(self) -> rpc_api.TrainingStatusResponse:
    """Get current skill training status."""
    ...


class TrainingStatusProvider(Protocol):
  """Protocol for objects that provide training status."""

  def get_training_status(self) -> rpc_api.ProgressTrainingStatusResponse:
    """Get current training status."""
    ...


# =============================================================================
# Shared styling constants and base class
# =============================================================================

# Apple color palette for consistent styling
APPLE_COLORS = {
    "blue": "#007aff",
    "green": "#34c759",
    "orange": "#ff9500",
    "red": "#ff3b30",
    "purple": "#af52de",
    "pink": "#ff2d55",
    "teal": "#5ac8fa",
    "indigo": "#5856d6",
    "yellow": "#ffcc00",
    "mint": "#00c7be",
    "cyan": "#32ade6",
}

# Apple-style phase colors
PHASE_COLORS = {
    "idle": "#86868b",
    "exporting": "#ff9500",
    "preparing_dataset": "#ff9500",
    "training": "#007aff",
    "finished": "#34c759",
    "failed": "#ff3b30",
}

# Phase display labels
PHASE_LABELS = {
    "idle": "Idle",
    "exporting": "Exporting",
    "preparing_dataset": "Preparing Dataset",
    "training": "Training",
    "finished": "Complete",
    "failed": "Failed",
}

# Common font family for Apple-style design
FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'SF Pro Display', "
    "'Segoe UI', Roboto, sans-serif"
)
TEXT_COLOR = "#1d1d1f"
SECONDARY_COLOR = "#86868b"


def _validate_color(color: str) -> str:
  """Validate and return accent color, with warning for invalid colors."""
  color_lower = color.lower()
  if color_lower not in APPLE_COLORS:
    import warnings

    valid = ", ".join(APPLE_COLORS.keys())
    warnings.warn(
        f"Unknown color '{color}'. Using 'blue'. Valid options: {valid}",
        stacklevel=3,
    )
    color_lower = "blue"
  return APPLE_COLORS[color_lower]


def _format_eta(seconds: float) -> str:
  """Format ETA in human-readable form."""
  if seconds < 60:
    return f"{seconds:.0f}s"
  elif seconds < 3600:
    return f"{seconds / 60:.1f}m"
  else:
    return f"{seconds / 3600:.1f}h"


class _BaseProgressWidget:
  """Base class for training progress widgets with shared functionality."""

  def __init__(self, poll_interval: float = 2.0, color: str = "blue") -> None:
    """Initialize base widget.

    Args:
        poll_interval: Seconds between status polls.
        color: Accent color for the widget.
    """
    self._poll_interval = poll_interval
    self._accent_color = _validate_color(color)
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

  def _get_style_dict(self) -> dict:
    """Return dict of common style values for widget templates."""
    return {
        "_font_family": FONT_FAMILY,
        "_text_color": TEXT_COLOR,
        "_secondary_color": SECONDARY_COLOR,
        "_accent_color": self._accent_color,
    }

  def _create_container_layout(self, width: str = "450px") -> Any:
    """Create modern card-style container layout."""
    return self._widgets_module.Layout(
        padding="20px",
        border="none",
        border_radius="12px",
        width=width,
        min_height="120px",
        background="#ffffff",
        box_shadow="0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05)",
        overflow="visible",
    )

  def _format_phase_html(self, phase: str, extra: str = "") -> str:
    """Format phase indicator HTML."""
    color = PHASE_COLORS.get(phase, "#86868b")
    label = PHASE_LABELS.get(phase, phase.title())
    html = (
        f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR}; font-weight: 500; letter-spacing: -0.01em;">'
        f'Phase: <span style="color: {color}; font-weight: 600;">'
        f"{label}</span>{extra}</div>"
    )
    return html

  def _format_error_html(self, error: str) -> str:
    """Format error message HTML."""
    return (
        f'<div style="font-family: {FONT_FAMILY}; font-size: 12px; '
        f"color: #ff3b30; background: #fff5f5; padding: 8px 12px; "
        f'border-radius: 6px; margin-top: 8px;">'
        f"⚠ {error}</div>"
    )

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

  def _create_widgets(self) -> dict:
    """Create widgets - must be overridden by subclasses."""
    raise NotImplementedError

  def _poll_loop(self) -> None:
    """Background polling loop - must be overridden by subclasses."""
    raise NotImplementedError


class TrainingProgressWidget(_BaseProgressWidget):
  """Interactive widget for monitoring training progress in Jupyter notebooks.

  Polls training status and displays progress with a clean, minimal UI.
  Uses ipywidgets for interactive display with modern Apple-style design.

  Example:
      trainer = ProgressPredictionTrainerClient(rpc_client)
      trainer.train_model(...)

      widget = TrainingProgressWidget(trainer, color="blue")
      widget.start()  # Shows live progress
      # Training completes or widget.stop() called
  """

  def __init__(
      self,
      status_provider: TrainingStatusProvider,
      poll_interval: float = 2.0,
      color: str = "blue",
  ) -> None:
    """Initialize the progress widget.

    Args:
        status_provider: Object with get_training_status() method.
        poll_interval: Seconds between status polls.
        color: Accent color for the widget. Options: blue, green, orange,
               red, purple, pink, teal, indigo, yellow, mint, cyan.
    """
    super().__init__(poll_interval=poll_interval, color=color)
    self._provider = status_provider

  def _create_widgets(self) -> dict:
    """Create the ipywidgets UI components with modern Apple-style design."""
    widgets = self._widgets_module

    # Phase indicator
    phase_label = widgets.HTML(
        value=self._format_phase_html("idle"),
        layout=widgets.Layout(margin="0 0 12px 0"),
    )

    # Progress bar with modern styling
    progress_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description="",
        bar_style="info",
        style={"bar_color": self._accent_color},
        layout=widgets.Layout(width="100%", height="6px"),
    )

    # Step counter with large modern typography
    step_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 24px; '
        f"color: {TEXT_COLOR}; font-weight: 600; letter-spacing: -0.02em; "
        'margin: 8px 0 4px 0;">0 / 0</div>'
    )

    # Train metrics display
    train_metrics_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
        "Train: Loss — &nbsp;•&nbsp; Acc — &nbsp;•&nbsp; F1 —</div>"
    )

    # Val metrics display
    val_metrics_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
        "Val: Loss — &nbsp;•&nbsp; Acc — &nbsp;•&nbsp; F1 —</div>"
    )

    # Speed and ETA
    speed_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR};">Speed: — &nbsp;•&nbsp; ETA: —</div>'
    )

    # Error display (hidden by default)
    error_label = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none", margin="8px 0 0 0"),
    )

    # Container with modern card styling
    container = widgets.VBox(
        [
            phase_label,
            progress_bar,
            step_label,
            train_metrics_label,
            val_metrics_label,
            speed_label,
            error_label,
        ],
        layout=self._create_container_layout(width="480px"),
    )

    result = {
        "container": container,
        "phase": phase_label,
        "progress": progress_bar,
        "steps": step_label,
        "train_metrics": train_metrics_label,
        "val_metrics": val_metrics_label,
        "speed": speed_label,
        "error": error_label,
    }
    result.update(self._get_style_dict())
    return result

  def _update_display(
      self, status: rpc_api.ProgressTrainingStatusResponse
  ) -> None:
    """Update widget display with current status."""
    if not self._widgets:
      return

    w = self._widgets
    phase = status.phase

    # Phase with optional checkpoint/model info
    extra = ""
    if status.checkpoint_id:
      extra += (
          f' <span style="color: {SECONDARY_COLOR}; font-weight: 400;">'
          f"• Checkpoint: {status.checkpoint_id}</span>"
      )
    if phase == "finished" and status.exported_model_id is not None:
      extra += (
          f' <span style="color: #34c759; font-weight: 500;">'
          f"• Model: {status.exported_model_id}</span>"
      )
    w["phase"].value = self._format_phase_html(phase, extra)

    # Update progress bar
    if status.max_steps > 0:
      progress_pct = (status.steps_completed / status.max_steps) * 100
      w["progress"].value = progress_pct

      # Change bar color based on phase
      if phase == "finished":
        w["progress"].style.bar_color = "#34c759"
      elif phase == "failed":
        w["progress"].style.bar_color = "#ff3b30"
      elif phase == "preparing_dataset":
        w["progress"].style.bar_color = "#ff9500"
      else:
        w["progress"].style.bar_color = self._accent_color
    else:
      w["progress"].value = 0

    # Update step counter with large modern typography
    w["steps"].value = (
        f'<div style="font-family: {FONT_FAMILY}; font-size: 24px; '
        f"color: {TEXT_COLOR}; font-weight: 600; letter-spacing: -0.02em; "
        f'margin: 8px 0 4px 0;">'
        f"{status.steps_completed:,} / {status.max_steps:,}</div>"
    )

    # Update train metrics
    loss_str = f"{status.loss:.4f}" if status.loss else "—"
    acc_str = f"{status.accuracy:.1%}" if status.accuracy is not None else "—"
    f1_str = f"{status.f1:.1%}" if status.f1 is not None else "—"
    w["train_metrics"].value = (
        f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
        f"Train: Loss {loss_str} &nbsp;•&nbsp; "
        f"Acc {acc_str} &nbsp;•&nbsp; F1 {f1_str}</div>"
    )

    # Update val metrics
    val_loss_str = (
        f"{status.val_loss:.4f}" if status.val_loss is not None else "—"
    )
    val_acc_str = (
        f"{status.val_accuracy:.1%}" if status.val_accuracy is not None else "—"
    )
    val_f1_str = f"{status.val_f1:.1%}" if status.val_f1 is not None else "—"
    w["val_metrics"].value = (
        f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
        f"Val: Loss {val_loss_str} &nbsp;•&nbsp; "
        f"Acc {val_acc_str} &nbsp;•&nbsp; F1 {val_f1_str}</div>"
    )

    # Update speed and ETA
    if status.fps and status.fps > 0:
      remaining_steps = status.max_steps - status.steps_completed
      eta_seconds = remaining_steps / status.fps if remaining_steps > 0 else 0
      eta_str = _format_eta(eta_seconds) if eta_seconds > 0 else "—"
      w["speed"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: {SECONDARY_COLOR};">'
          f"Speed: {status.fps:.1f} steps/s &nbsp;•&nbsp; ETA: {eta_str}</div>"
      )
    else:
      w["speed"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: {SECONDARY_COLOR};">Speed: calculating...</div>'
      )

    # Update error display
    if status.error:
      w["error"].value = self._format_error_html(status.error)
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


class SkillTrainingProgressWidget(_BaseProgressWidget):
  """Interactive widget for monitoring skill training progress in Jupyter notebooks.

  Handles both dataset export and training phases with appropriate visualizations.
  Uses ipywidgets for interactive display with fallback to console output.

  Example:
      trainer = TrainerClient(rpc_client)
      trainer.train_skill_model(...)

      widget = SkillTrainingProgressWidget(trainer)
      widget.start()  # Shows live progress
      widget.wait()   # Wait for completion
  """

  def __init__(
      self,
      status_provider: SkillTrainingStatusProvider,
      poll_interval: float = 2.0,
      color: str = "blue",
  ) -> None:
    """Initialize the skill training progress widget.

    Args:
        status_provider: Object with get_training_status() method.
        poll_interval: Seconds between status polls.
        color: Accent color for the widget. Options: blue, green, orange,
               red, purple, pink, teal, indigo, yellow, mint, cyan.
    """
    super().__init__(poll_interval=poll_interval, color=color)
    self._provider = status_provider
    self._matplotlib_available = False

    # Loss history for plotting
    self._steps_history: list[int] = []
    self._loss_history: list[float] = []
    self._max_history = 500  # Keep last N points

    # Try to import matplotlib
    try:
      import matplotlib.pyplot as plt

      self._plt = plt
      self._matplotlib_available = True
    except ImportError:
      pass

  def _create_widgets(self) -> dict:
    """Create the ipywidgets UI components with modern Apple-style design."""
    widgets = self._widgets_module

    # Phase indicator
    phase_label = widgets.HTML(
        value=self._format_phase_html("idle"),
        layout=widgets.Layout(margin="0 0 12px 0"),
    )

    # Progress bar with modern styling
    progress_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description="",
        bar_style="info",
        style={"bar_color": self._accent_color},
        layout=widgets.Layout(width="100%", height="6px"),
    )

    # Progress details (steps or export entries)
    progress_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 24px; '
        f"color: {TEXT_COLOR}; font-weight: 600; letter-spacing: -0.02em; "
        'margin: 8px 0 4px 0;">0 / 0</div>'
    )

    # Metrics display (loss, fps)
    metrics_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
        "Loss: — &nbsp;•&nbsp; Speed: —</div>"
    )

    # ETA display
    eta_label = widgets.HTML(
        value=f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
        f'color: {SECONDARY_COLOR};">ETA: —</div>'
    )

    # Error display (hidden by default)
    error_label = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none", margin="8px 0 0 0"),
    )

    # Loss plot as HTML (hidden until training starts)
    loss_plot_html = widgets.HTML(
        value="",
        layout=widgets.Layout(
            width="100%",
            display="none",
            margin="12px 0 0 0",
        ),
    )

    # Container with modern card styling
    container = widgets.VBox(
        [
            phase_label,
            progress_bar,
            progress_label,
            metrics_label,
            eta_label,
            loss_plot_html,
            error_label,
        ],
        layout=self._create_container_layout(width="450px"),
    )

    result = {
        "container": container,
        "phase": phase_label,
        "progress": progress_bar,
        "progress_label": progress_label,
        "metrics": metrics_label,
        "eta": eta_label,
        "loss_plot": loss_plot_html,
        "error": error_label,
    }
    result.update(self._get_style_dict())
    return result

  def _update_loss_plot(self) -> None:
    """Update the loss plot with current history."""
    if not self._widgets or not self._matplotlib_available:
      return
    if len(self._loss_history) < 2:
      return

    w = self._widgets

    # Use object-oriented API to avoid pyplot auto-display issues
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib as mpl
    import io
    import base64

    # Set font settings before creating figure
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["axes.labelcolor"] = "#86868b"
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["xtick.color"] = "#86868b"
    mpl.rcParams["ytick.color"] = "#86868b"
    mpl.rcParams["text.color"] = "#86868b"
    mpl.rcParams["mathtext.default"] = "regular"

    # Filter out zero/near-zero values from beginning (before training starts)
    steps_filtered = []
    loss_filtered = []
    for s, l in zip(self._steps_history, self._loss_history):
      if l > 1e-10:  # Skip essentially-zero values
        steps_filtered.append(s)
        loss_filtered.append(l)

    if len(loss_filtered) < 2:
      return  # Not enough valid data to plot

    # Smooth the loss values using exponential moving average
    def smooth_ema(values, alpha=0.3):
      """Exponential moving average smoothing."""
      smoothed = [values[0]]
      for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
      return smoothed

    loss_smoothed = smooth_ema(loss_filtered, alpha=0.3)

    # Get accent color
    accent = self._accent_color

    # Create figure using OO API (no pyplot state machine)
    fig = Figure(figsize=(4.0, 1.5), facecolor="#ffffff")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")

    # Plot smoothed line with gradient fill
    ax.plot(
        steps_filtered,
        loss_smoothed,
        color=accent,
        linewidth=2,
        solid_capstyle="round",
    )
    ax.fill_between(
        steps_filtered,
        loss_smoothed,
        alpha=0.1,
        color=accent,
    )

    # Floating look - hide all spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Subtle horizontal grid only
    ax.grid(True, axis="y", alpha=0.1, color="#86868b", linewidth=0.5)
    ax.set_axisbelow(True)

    # Log scale
    ax.set_yscale("log")
    ax.yaxis.set_visible(False)

    # X-axis ticks - minimal
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.tick_params(axis="x", labelsize=8, colors="#86868b", length=0, pad=6)

    # Get loss values for labels (actual values, but position on smoothed curve)
    max_loss = max(loss_filtered)
    max_idx = loss_filtered.index(max_loss)
    max_loss_step = steps_filtered[max_idx]
    max_loss_smoothed = loss_smoothed[max_idx]  # Position on smoothed curve

    current_loss = loss_filtered[-1]
    current_loss_smoothed = loss_smoothed[-1]  # Position on smoothed curve
    current_step = steps_filtered[-1]

    def fmt_loss(x):
      if x >= 1:
        return f"{x:.2f}"
      elif x >= 0.01:
        return f"{x:.3f}"
      else:
        return f"{x:.4f}"

    # Add floating text labels on the graph
    # Max loss label (positioned above the smoothed curve point)
    ax.annotate(
        f"{fmt_loss(max_loss)}  (max, step {max_loss_step:,})",
        xy=(max_loss_step, max_loss_smoothed),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=7,
        color="#86868b",
        ha="left",
        va="bottom",
    )

    # Current loss label (positioned on the smoothed curve, with background)
    ax.annotate(
        f" {fmt_loss(current_loss)} ",
        xy=(current_step, current_loss_smoothed),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8,
        color=accent,
        ha="left",
        va="center",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
    )

    # Tight layout
    fig.tight_layout(pad=0.5)

    # Render to PNG bytes and encode as base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Check if training was resumed (first recorded step > 0)
    first_step = steps_filtered[0] if steps_filtered else 0
    caption = ""
    if first_step > 0:
      caption = (
          f'<div style="font-size: 10px; color: #86868b; margin-top: 4px; '
          f'font-style: italic;">Showing from step {first_step:,}</div>'
      )

    # Set HTML value with image and optional caption
    w["loss_plot"].value = (
        f'<img src="data:image/png;base64,{img_base64}" '
        f'style="width: 100%; max-width: 400px;" />{caption}'
    )

  def _update_display(self, status: rpc_api.TrainingStatusResponse) -> None:
    """Update widget display with current status."""
    if not self._widgets:
      return

    w = self._widgets
    phase = status.phase

    # Update phase indicator
    w["phase"].value = self._format_phase_html(phase)

    # Handle different phases
    if phase in ("exporting", "preparing_dataset"):
      # Dataset export/preparation phase
      total = max(status.export_entries_total, 1)
      processed = status.export_entries_processed
      progress_pct = (processed / total) * 100

      w["progress"].value = progress_pct
      w["progress"].style.bar_color = "#ff9500"
      w["progress_label"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 24px; '
          f"color: {TEXT_COLOR}; font-weight: 600; letter-spacing: -0.02em; "
          f'margin: 8px 0 4px 0;">{processed:,} / {total:,}</div>'
      )
      w["metrics"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
          "Preparing dataset...</div>"
      )
      w["eta"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: {SECONDARY_COLOR};">Speed: calculating...</div>'
      )
      # Hide loss plot during export
      w["loss_plot"].layout.display = "none"

    elif phase == "training":
      # Training phase
      if status.max_steps > 0:
        progress_pct = (status.steps_completed / status.max_steps) * 100
        w["progress"].value = progress_pct
      else:
        w["progress"].value = 0

      w["progress"].style.bar_color = self._accent_color

      # Steps with loss on the right
      loss_str = f"{status.loss:.6f}" if status.loss is not None else "—"
      w["progress_label"].value = (
          f'<div style="font-family: {FONT_FAMILY}; display: flex; '
          f"justify-content: space-between; align-items: baseline; "
          f'margin: 8px 0 4px 0;">'
          f'<span style="font-size: 24px; color: {TEXT_COLOR}; font-weight: 600; '
          f'letter-spacing: -0.02em;">'
          f"{status.steps_completed:,} / {status.max_steps:,}</span>"
          f'<span style="font-size: 14px; color: {SECONDARY_COLOR}; '
          f'font-weight: 500;">Loss: {loss_str}</span></div>'
      )

      # Speed only in metrics now
      if status.fps is not None and status.fps > 0:
        fps_str = f"{status.fps:.1f} steps/s"
      else:
        fps_str = "calculating..."
      w["metrics"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: {SECONDARY_COLOR}; margin: 4px 0;">'
          f"Speed: {fps_str}</div>"
      )

      # ETA
      if (
          status.fps is not None
          and status.fps > 0
          and status.max_steps > status.steps_completed
      ):
        remaining = status.max_steps - status.steps_completed
        eta_seconds = remaining / status.fps
        eta_str = _format_eta(eta_seconds)
        w["eta"].value = (
            f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
            f'color: {SECONDARY_COLOR};">ETA: {eta_str}</div>'
        )
      else:
        w["eta"].value = (
            f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
            f'color: {SECONDARY_COLOR};">ETA: calculating...</div>'
        )

      # Update loss plot (only when loss value changes)
      if status.loss is not None and self._matplotlib_available:
        # Only add new point if loss changed or it's the first point
        should_add = (
            len(self._loss_history) == 0
            or status.loss != self._loss_history[-1]
            or status.steps_completed != self._steps_history[-1]
        )

        if should_add:
          self._steps_history.append(status.steps_completed)
          self._loss_history.append(status.loss)

          # Trim history if needed
          if len(self._steps_history) > self._max_history:
            self._steps_history = self._steps_history[-self._max_history :]
            self._loss_history = self._loss_history[-self._max_history :]

        # Show plot widget
        w["loss_plot"].layout.display = "block"

        # Update the plot
        self._update_loss_plot()

    elif phase == "finished":
      w["progress"].value = 100
      w["progress"].style.bar_color = "#34c759"
      loss_str = f"{status.loss:.6f}" if status.loss is not None else "—"
      w["progress_label"].value = (
          f'<div style="font-family: {FONT_FAMILY}; display: flex; '
          f"justify-content: space-between; align-items: baseline; "
          f'margin: 8px 0 4px 0;">'
          f'<span style="font-size: 24px; color: {TEXT_COLOR}; font-weight: 600; '
          f'letter-spacing: -0.02em;">'
          f"{status.steps_completed:,} / {status.max_steps:,}</span>"
          f'<span style="font-size: 14px; color: #34c759; font-weight: 500;">'
          f"Final Loss: {loss_str}</span></div>"
      )
      w["metrics"].value = ""
      w["eta"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: #34c759; font-weight: 500;">✓ Training complete</div>'
      )
      # Keep loss plot visible with final state
      if self._loss_history and self._matplotlib_available:
        w["loss_plot"].layout.display = "block"

    elif phase == "failed":
      w["progress"].style.bar_color = "#ff3b30"
      w["metrics"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: #ff3b30; margin: 4px 0;">Training failed</div>'
      )
      w["eta"].value = ""

    else:
      # Idle or other states
      w["progress"].value = 0
      w["progress_label"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 24px; '
          f"color: {TEXT_COLOR}; font-weight: 600; letter-spacing: -0.02em; "
          f'margin: 8px 0 4px 0;">— / —</div>'
      )
      w["metrics"].value = (
          f'<div style="font-family: {FONT_FAMILY}; font-size: 13px; '
          f'color: {SECONDARY_COLOR}; margin: 4px 0;">Waiting to start...</div>'
      )
      w["eta"].value = ""

    # Error display
    if status.metrics and "error" in status.metrics:
      error_msg = str(status.metrics["error"])
      w["error"].value = self._format_error_html(error_msg)
      w["error"].layout.display = "block"
    else:
      w["error"].layout.display = "none"

  def _print_status(self, status: rpc_api.TrainingStatusResponse) -> None:
    """Print status to console (fallback when widgets unavailable)."""
    phase = status.phase

    if phase in ("exporting", "preparing_dataset"):
      total = max(status.export_entries_total, 1)
      progress_pct = (status.export_entries_processed / total) * 100
      bar_width = 30
      filled = int(bar_width * progress_pct / 100)
      bar = "=" * filled + "-" * (bar_width - filled)
      print(
          f"[{phase}] [{bar}] {progress_pct:.1f}% "
          f"({status.export_entries_processed}/{status.export_entries_total} entries)"
      )

    elif phase in ("training", "finished"):
      progress_pct = (
          (status.steps_completed / status.max_steps * 100)
          if status.max_steps > 0
          else 0
      )
      bar_width = 30
      filled = int(bar_width * progress_pct / 100)
      bar = "=" * filled + "-" * (bar_width - filled)

      metrics = []
      if status.loss is not None:
        metrics.append(f"loss={status.loss:.6f}")
      if status.fps is not None and status.fps > 0:
        metrics.append(f"fps={status.fps:.2f}")
      metrics_str = (
          ", ".join(metrics) if metrics else "(waiting for metrics...)"
      )

      eta_str = ""
      if (
          status.fps is not None
          and status.fps > 0
          and status.max_steps > status.steps_completed
      ):
        remaining = status.max_steps - status.steps_completed
        eta_seconds = remaining / status.fps
        eta_str = f" | ETA: {_format_eta(eta_seconds)}"

      print(
          f"[{phase}] [{bar}] {progress_pct:.1f}% "
          f"({status.steps_completed}/{status.max_steps}) {metrics_str}{eta_str}"
      )

    else:
      print(f"[{phase}] Waiting...")

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
          self._widgets["error"].value = self._format_error_html(str(e))
          self._widgets["error"].layout.display = "block"
        else:
          print(f"Poll error: {e}")

      self._stop_event.wait(self._poll_interval)

  def start(self) -> None:
    """Start displaying progress.

    Creates widgets (if in notebook) and begins polling in background.
    Clears loss history for fresh tracking.
    """
    # Clear loss history for fresh tracking
    self._steps_history = []
    self._loss_history = []
    super().start()


def monitor_skill_training(
    status_provider: SkillTrainingStatusProvider,
    poll_interval: float = 2.0,
    timeout: float | None = None,
) -> rpc_api.TrainingStatusResponse:
  """Convenience function to monitor skill training until completion.

  Creates a widget, displays it, and waits for training to finish.

  Args:
      status_provider: Object with get_training_status() method (TrainerClient).
      poll_interval: Seconds between status polls.
      timeout: Maximum seconds to wait, or None for unlimited.

  Returns:
      Final training status.

  Example:
      trainer.train_skill_model(...)
      final_status = monitor_skill_training(trainer)
      print(f"Final loss: {final_status.loss}")
  """
  widget = SkillTrainingProgressWidget(status_provider, poll_interval)
  widget.start()
  widget.wait(timeout)
  widget.stop()
  return status_provider.get_training_status()


class SkillTrainingUI:
  """Complete interactive UI for skill training in Jupyter notebooks.

  Provides a single widget with all controls - no code needed after setup.

  Usage:
      from r2_labs.sdk.progress_widget import SkillTrainingUI

      ui = SkillTrainingUI(trainer)
      ui.display()  # That's it! Everything else is clickable.
  """

  def __init__(self, trainer: Any) -> None:
    """Initialize the training UI.

    Args:
        trainer: TrainerClient instance connected to the training server.
    """
    self._trainer = trainer
    self._widgets: dict | None = None
    self._progress_widget: SkillTrainingProgressWidget | None = None
    self._is_training = False

    # Check for ipywidgets
    try:
      import ipywidgets as widgets
      from IPython.display import display

      self._widgets_module = widgets
      self._display = display
      self._available = True
    except ImportError:
      self._available = False

  def _create_ui(self) -> None:
    """Create the complete UI."""
    if not self._available:
      print("ipywidgets not available. Install with: pip install ipywidgets")
      return

    widgets = self._widgets_module

    # Styling (uses shared constants)
    label_style = (
        f"font-family: {FONT_FAMILY}; font-size: 12px; color: {SECONDARY_COLOR}; "
        "font-weight: 500; margin-bottom: 4px;"
    )
    input_layout = widgets.Layout(width="100%")

    # === Configuration Section ===
    # Model name
    model_name_label = widgets.HTML(
        f'<div style="{label_style}">Model Name</div>'
    )
    self._model_name = widgets.Text(
        value="my_skill_model",
        placeholder="e.g., pick_and_place",
        layout=input_layout,
    )

    # Entry filter
    entry_filter_label = widgets.HTML(
        f'<div style="{label_style}">Entry Filter (glob pattern)</div>'
    )
    self._entry_filter = widgets.Text(
        value="*",
        placeholder="e.g., pick_*",
        layout=input_layout,
    )

    # Training steps
    steps_label = widgets.HTML(
        f'<div style="{label_style}">Training Steps</div>'
    )
    self._training_steps = widgets.IntSlider(
        value=10000,
        min=1000,
        max=100000,
        step=1000,
        readout=True,
        layout=input_layout,
    )

    # Batch size
    batch_label = widgets.HTML(f'<div style="{label_style}">Batch Size</div>')
    self._batch_size = widgets.Dropdown(
        options=[8, 16, 32, 64, 128],
        value=32,
        layout=widgets.Layout(width="100px"),
    )

    # Prediction horizon
    horizon_label = widgets.HTML(
        f'<div style="{label_style}">Prediction Horizon</div>'
    )
    self._pred_horizon = widgets.Dropdown(
        options=[8, 16, 32, 64],
        value=32,
        layout=widgets.Layout(width="100px"),
    )

    # Force rebuild checkbox
    self._force_rebuild = widgets.Checkbox(
        value=False,
        description="Force rebuild dataset",
        indent=False,
    )

    # === Action Buttons ===
    button_style = widgets.Layout(width="120px", height="36px")

    self._start_btn = widgets.Button(
        description="▶ Start",
        button_style="primary",
        layout=button_style,
    )
    self._start_btn.on_click(self._on_start)

    self._cancel_btn = widgets.Button(
        description="◼ Cancel",
        button_style="warning",
        layout=button_style,
        disabled=True,
    )
    self._cancel_btn.on_click(self._on_cancel)

    self._export_btn = widgets.Button(
        description="↗ Export",
        button_style="success",
        layout=button_style,
        disabled=True,
    )
    self._export_btn.on_click(self._on_export)

    # === Status/Output Area ===
    self._status_output = widgets.Output(
        layout=widgets.Layout(width="100%", min_height="40px")
    )

    # === Progress Widget Container ===
    self._progress_container = widgets.Output(
        layout=widgets.Layout(width="100%")
    )

    # === Layout ===
    config_section = widgets.VBox(
        [
            model_name_label,
            self._model_name,
            widgets.HTML('<div style="height: 8px;"></div>'),
            entry_filter_label,
            self._entry_filter,
            widgets.HTML('<div style="height: 8px;"></div>'),
            steps_label,
            self._training_steps,
            widgets.HTML('<div style="height: 8px;"></div>'),
            widgets.HBox(
                [
                    widgets.VBox([batch_label, self._batch_size]),
                    widgets.HTML('<div style="width: 20px;"></div>'),
                    widgets.VBox([horizon_label, self._pred_horizon]),
                ],
            ),
            widgets.HTML('<div style="height: 8px;"></div>'),
            self._force_rebuild,
        ],
        layout=widgets.Layout(padding="0 0 16px 0"),
    )

    buttons_section = widgets.HBox(
        [self._start_btn, self._cancel_btn, self._export_btn],
        layout=widgets.Layout(gap="8px", margin="0 0 16px 0"),
    )

    # Main container
    self._container = widgets.VBox(
        [
            widgets.HTML(
                f'<div style="font-family: {FONT_FAMILY}; font-size: 18px; '
                f'font-weight: 600; color: {TEXT_COLOR}; margin-bottom: 16px;">'
                "Skill Training</div>"
            ),
            config_section,
            buttons_section,
            self._status_output,
            self._progress_container,
        ],
        layout=widgets.Layout(
            padding="24px",
            border="none",
            border_radius="12px",
            width="480px",
            background="#ffffff",
            box_shadow="0 1px 3px rgba(0,0,0,0.08), "
            "0 4px 12px rgba(0,0,0,0.05)",
        ),
    )

  def _show_status(self, message: str, is_error: bool = False) -> None:
    """Show a status message."""
    color = "#ff3b30" if is_error else "#86868b"
    self._status_output.clear_output()
    with self._status_output:
      from IPython.display import HTML, display

      display(
          HTML(
              f'<div style="font-size: 13px; color: {color}; '
              f'padding: 8px 0;">{message}</div>'
          )
      )

  def _on_start(self, _) -> None:
    """Handle start button click."""
    if self._is_training:
      return

    self._is_training = True
    self._start_btn.disabled = True
    self._cancel_btn.disabled = False
    self._export_btn.disabled = True

    # Clear previous progress
    self._progress_container.clear_output()

    try:
      # Start training
      response = self._trainer.train_skill_model(
          model_name=self._model_name.value,
          training_steps=self._training_steps.value,
          entry_filter=self._entry_filter.value,
          batch_size=self._batch_size.value,
          prediction_horizon=self._pred_horizon.value,
          force_rebuild=self._force_rebuild.value,
          timeout=600000,
      )

      if response.error:
        self._show_status(f"Error: {response.error}", is_error=True)
        self._is_training = False
        self._start_btn.disabled = False
        self._cancel_btn.disabled = True
        return

      self._show_status("Training started...")

      # Show progress widget
      with self._progress_container:
        self._progress_widget = SkillTrainingProgressWidget(
            self._trainer, poll_interval=2.0
        )
        self._progress_widget.start()

      # Monitor in background
      import threading

      def monitor():
        if self._progress_widget:
          self._progress_widget.wait()
        self._on_training_complete()

      threading.Thread(target=monitor, daemon=True).start()

    except Exception as e:
      self._show_status(f"Error: {e}", is_error=True)
      self._is_training = False
      self._start_btn.disabled = False
      self._cancel_btn.disabled = True

  def _on_training_complete(self) -> None:
    """Called when training completes."""
    self._is_training = False
    self._start_btn.disabled = False
    self._cancel_btn.disabled = True
    self._export_btn.disabled = False

    status = self._trainer.get_training_status()
    if status.phase == "finished":
      self._show_status("✓ Training complete! Click Export to save model.")
    elif status.phase == "failed":
      self._show_status("Training failed.", is_error=True)

  def _on_cancel(self, _) -> None:
    """Handle cancel button click."""
    try:
      response = self._trainer.cancel_training(export_model=False)
      if response.success:
        self._show_status("Training cancelled.")
        if self._progress_widget:
          self._progress_widget.stop()
      else:
        self._show_status(f"Cancel failed: {response.error}", is_error=True)
    except Exception as e:
      self._show_status(f"Error: {e}", is_error=True)

    self._is_training = False
    self._start_btn.disabled = False
    self._cancel_btn.disabled = True

  def _on_export(self, _) -> None:
    """Handle export button click."""
    try:
      self._show_status("Exporting model...")
      response = self._trainer.export_model()
      if response.success:
        self._show_status(f"✓ Model exported: {response.model_id}")
      else:
        self._show_status(f"Export failed: {response.error}", is_error=True)
    except Exception as e:
      self._show_status(f"Error: {e}", is_error=True)

  def display(self) -> None:
    """Display the training UI."""
    if not self._available:
      print("ipywidgets not available. Install with: pip install ipywidgets")
      return

    self._create_ui()
    self._display(self._container)
