"""Dash UI for episode collection using the r2_labs SDK.

Example to run:

uv run python r2_labs/examples/scripts/collect_data.py \
  --robot_hostname=akhilraju-home.local \
  --enable_pedal \
  --entry_prefix=test_rectify_open_latch

Example with alternating prefixes (switches on each save/discard):

uv run python r2_labs/examples/scripts/collect_data.py \
  --robot_hostname=akhilraju-home.local \
  --enable_pedal \
  --entry_prefix=task_grasp,task_place

Example with continuous teleop and no start trajectory:

uv run python r2_labs/examples/scripts/collect_data.py \
  --robot_hostname=akhilraju-home.local \
  --enable_pedal \
  --entry_prefix=test_task \
  --continuous_teleop \
  --start_trajectory=None

Dependencies (pip):
  - absl-py
  - dash
  - evdev (only required when --enable_pedal is set)
  - r2_labs (from this repo or installed package)
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import signal
import threading
import time

import dash
import dotenv
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from loguru import logger as log

try:
  import evdev
  from evdev import InputDevice, ecodes
except ImportError:  # pragma: no cover - optional dependency.
  evdev = None
  InputDevice = None
  ecodes = None

from absl import app as absl_app
from absl import flags

from r2_labs import client as r2client
from r2_labs import rpc_api
from r2_labs.sdk import logging as r2_logging
from r2_labs.sdk import sentry

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "robot_hostname",
    "localhost",
    "Hostname of the robot backend.",
)
flags.DEFINE_integer(
    "robot_port",
    rpc_api.DEFAULT_PORT,
    "Main RPC port of the robot backend.",
)
flags.DEFINE_integer(
    "robot_query_port",
    rpc_api.DEFAULT_QUERY_PORT,
    "Query RPC port of the robot backend.",
)
flags.DEFINE_string(
    "web_host",
    "0.0.0.0",
    "Host/IP for the Dash web server.",
)
flags.DEFINE_integer(
    "web_port",
    8050,
    "Port for the Dash web server.",
)
flags.DEFINE_integer(
    "poll_interval_ms",
    1000,
    "Polling interval for episode state updates.",
)
flags.DEFINE_bool(
    "enable_pedal",
    False,
    "Enable USB foot pedal control on this client.",
)
flags.DEFINE_string(
    "pedal_device_path",
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd",
    "Device path for the foot pedal.",
)
flags.DEFINE_list(
    "entry_prefix",
    None,
    "Entry prefix(es) for saved episodes. Pass two comma-separated "
    "prefixes (e.g. 'a,b') to alternate between them on each save/discard.",
)
flags.DEFINE_bool(
    "enable_camera_vis",
    True,
    "Enable live camera visualization in the UI.",
)
flags.DEFINE_bool(
    "continuous_teleop",
    False,
    "If True, teleop continues during reset. If False, teleop is disabled "
    "during reset while the robot moves to the start position.",
)
flags.DEFINE_string(
    "start_trajectory",
    "Pre-insert motion Rectify",
    "Name of the trajectory to move to at episode start. Set to 'None' to skip.",
)


def episode_reset(
    robot: r2client.Robot,
    start_event: threading.Event,
    ready_event: threading.Event,
    waiting_event: threading.Event,
    ready_for_start_event: threading.Event,
    continuous_teleop: bool = False,
    start_trajectory: str | None = None,
) -> None:
  """Called at the beginning of each episode to reset things.

  You can update this function to put in your own reset logic.

  Args:
    robot: The robot client.
    start_event: Event to signal episode start.
    ready_event: Event to signal reset is complete.
    waiting_event: Event to signal waiting for start.
    ready_for_start_event: Event to signal ready for start button.
    continuous_teleop: If True, teleop continues during reset.
    start_trajectory: Name of trajectory to move to. None to skip.
  """

  # Add your reset logic here.

  # Set execution mode based on continuous_teleop setting.
  if continuous_teleop:
    robot.exec_mode.set_execution_mode(
        rpc_api.ExecutionMode.DATA_COLLECTION_TELEOP
    )
  else:
    robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)

  # Move to start trajectory if specified.
  if start_trajectory:
    motion_future = robot.behaviour.trajectory_motion(
        trajectory_name=start_trajectory,
        motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
        static_gripper=False,
        period_seconds=None,
    )
    print(f"Moving to reset pose ({start_trajectory})...")
    motion_future.result()
  else:
    print("Skipping trajectory motion (start_trajectory=None).")

  print("Aligning leader arm with follower...")

  motion_future = robot.behaviour.align_leader_with_follower(
      timeout_seconds=3.0,
      threshold=0.1,
  )
  motion_future.result()

  # This must be the last block, to make sure the robot is in the right mode for
  # teleop data collection.
  ready_for_start_event.set()
  waiting_event.set()
  start_event.wait()
  start_event.clear()
  waiting_event.clear()
  robot.exec_mode.set_execution_mode(
      rpc_api.ExecutionMode.DATA_COLLECTION_TELEOP
  )
  ready_event.set()


class Button(enum.Enum):
  """Enum for pedal buttons."""

  A = 0
  B = 1
  C = 2
  OTHER = 3

  @classmethod
  def from_evdev_code(cls, code: int) -> "Button":
    if ecodes is None:
      raise RuntimeError("evdev is required for pedal support.")
    if code == ecodes.KEY_A:
      return cls.A
    if code == ecodes.KEY_B:
      return cls.B
    if code == ecodes.KEY_C:
      return cls.C
    return cls.OTHER


class ButtonState(enum.Enum):
  """Enum for pedal button states."""

  RELEASED = 0
  PRESSED = 1
  HOLD = 2

  @classmethod
  def from_evdev_value(cls, value: int) -> "ButtonState":
    if value == 0:
      return cls.RELEASED
    if value == 1:
      return cls.PRESSED
    if value == 2:
      return cls.HOLD
    raise ValueError(f"Invalid evdev value: {value}")


@dataclasses.dataclass
class Event:
  """Dataclass for pedal events."""

  button: Button
  state: ButtonState

  @classmethod
  def from_evdev_event(cls, event) -> "Event":
    button = Button.from_evdev_code(event.code)
    state = ButtonState.from_evdev_value(event.value)
    return cls(button=button, state=state)

  @classmethod
  def is_pedal_event(cls, event) -> bool:
    if ecodes is None:
      raise RuntimeError("evdev is required for pedal support.")
    return event.type in (ecodes.EV_KEY,)


_PEDAL_MAX_RECONNECT_ATTEMPTS = 30
_PEDAL_RECONNECT_BASE_DELAY_SEC = 0.5
_PEDAL_RECONNECT_MAX_DELAY_SEC = 3.0


class PedalListener:
  """Listen to a 3-button foot pedal via evdev.

  Automatically reconnects on USB device disconnection with exponential
  backoff, so transient cable or hub issues don't kill data collection.
  """

  def __init__(self, device_path: str, on_click):
    self._device_path = device_path
    self._device = None
    self._thread: threading.Thread | None = None
    self._running = False
    self._on_click = on_click

  def _open_device(self):
    if InputDevice is None:
      raise RuntimeError("evdev is required for pedal support.")
    try:
      self._device = InputDevice(self._device_path)
    except FileNotFoundError as fnfe:
      raise ValueError(
          f"Device not found at {self._device_path}. Please check the path."
      ) from fnfe

  def _close_device(self):
    if self._device is not None:
      try:
        self._device.close()
      except OSError:
        pass
      self._device = None

  def _reconnect(self) -> bool:
    """Try to reopen the device with exponential backoff."""
    for attempt in range(_PEDAL_MAX_RECONNECT_ATTEMPTS):
      if not self._running:
        return False
      delay = min(
          _PEDAL_RECONNECT_BASE_DELAY_SEC * (2 ** min(attempt, 3)),
          _PEDAL_RECONNECT_MAX_DELAY_SEC,
      )
      # sleep in short intervals so stop() isn't blocked long
      deadline = time.monotonic() + delay
      while time.monotonic() < deadline and self._running:
        time.sleep(0.1)
      if not self._running:
        return False
      try:
        self._open_device()
        return True
      except (OSError, ValueError):
        log.debug(
            "Pedal reconnect attempt {}/{} failed.",
            attempt + 1,
            _PEDAL_MAX_RECONNECT_ATTEMPTS,
        )
    return False

  def _event_loop(self):
    assert self._device is not None, "Device must be opened first"
    while self._running:
      try:
        for event in self._device.read_loop():
          if not self._running:
            return
          if Event.is_pedal_event(event):
            pedal_event = Event.from_evdev_event(event)
            try:
              self._on_click(pedal_event)
            except Exception:
              log.exception("Pedal event handler failed")
      except OSError:
        if not self._running:
          return
        log.warning(
            "Pedal device lost ({}), attempting to reconnect...",
            self._device_path,
        )
        self._close_device()
        if self._reconnect():
          log.info("Pedal device reconnected ({}).", self._device_path)
        else:
          log.error("Pedal reconnection failed, giving up.")
          return

  def start(self):
    if self._running:
      return
    self._running = True
    self._open_device()
    self._thread = threading.Thread(target=self._event_loop, daemon=True)
    self._thread.start()

  def stop(self):
    if not self._running:
      return
    self._running = False
    # close device first to unblock any in-progress read_loop() call
    self._close_device()
    if self._thread is not None:
      self._thread.join(timeout=5.0)
      self._thread = None


class ResetCoordinator:
  """Coordinate episode resets and start triggers."""

  def __init__(
      self,
      robot: r2client.Robot,
      continuous_teleop: bool = False,
      start_trajectory: str | None = None,
  ) -> None:
    self._robot = robot
    self._continuous_teleop = continuous_teleop
    self._start_trajectory = start_trajectory
    self._reset_requested = threading.Event()
    self._start_requested = threading.Event()
    self._ready = threading.Event()
    self._waiting_for_start = threading.Event()
    self._ready_for_start = threading.Event()
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self) -> None:
    while not self._stop.is_set():
      self._reset_requested.wait()
      if self._stop.is_set():
        break
      self._reset_requested.clear()
      try:
        episode_reset(
            self._robot,
            self._start_requested,
            self._ready,
            self._waiting_for_start,
            self._ready_for_start,
            continuous_teleop=self._continuous_teleop,
            start_trajectory=self._start_trajectory,
        )
      except Exception:
        log.exception("episode_reset failed, will retry on next request")

  def request_reset(self) -> None:
    self._ready.clear()
    self._ready_for_start.clear()
    self._waiting_for_start.clear()
    self._start_requested.clear()
    self._reset_requested.set()

  def request_start(self) -> bool:
    if not self._ready_for_start.is_set():
      return False
    self._ready_for_start.clear()
    self._waiting_for_start.clear()
    self._start_requested.set()
    return True

  def allow_start_retry(self) -> None:
    self._ready_for_start.set()
    self._waiting_for_start.clear()
    self._start_requested.clear()

  def wait_until_ready(self, timeout: float | None = None) -> bool:
    return self._ready.wait(timeout=timeout)

  def is_waiting_for_start(self) -> bool:
    return self._waiting_for_start.is_set()

  def is_ready_for_start(self) -> bool:
    return self._ready_for_start.is_set()

  def stop(self) -> None:
    self._stop.set()
    self._reset_requested.set()
    self._thread.join(timeout=1.0)


class EpisodeController:
  """Thread-safe wrapper around EpisodeObserverClient."""

  def __init__(
      self,
      episode_client: r2client.EpisodeObserverClient,
      reset_coordinator: "ResetCoordinator",
      prefixes: list[str],
  ) -> None:
    self._episode_client = episode_client
    self._reset_coordinator = reset_coordinator
    self._prefixes = list(prefixes)
    self._prefix_index = 0
    self._lock = threading.Lock()
    self._saved_count = 0
    self._discarded_count = 0

  def start(self) -> None:
    if not self._reset_coordinator.request_start():
      raise RuntimeError("Not ready to start episode (reset not complete).")
    if not self._reset_coordinator.wait_until_ready(timeout=30.0):
      # Timed out — trigger a fresh reset so the system can recover.
      self._reset_coordinator.request_reset()
      raise RuntimeError("Timed out waiting for episode reset to complete.")
    try:
      with self._lock:
        self._episode_client.start()
    except Exception:
      self._reset_coordinator.allow_start_retry()
      raise

  def stop(self) -> None:
    with self._lock:
      self._episode_client.stop()
    self._reset_coordinator.request_reset()

  @property
  def is_alternating(self) -> bool:
    return len(self._prefixes) == 2

  def save(self) -> None:
    with self._lock:
      prefix = self._prefixes[self._prefix_index] if self._prefixes else None
      if prefix is None:
        raise ValueError("entry_prefix is required before saving.")
      self._episode_client.save(entry_prefix=prefix)
      self._saved_count += 1
      log.info("Saved episodes: {}", self._saved_count)
      self._advance_prefix()

  def set_entry_prefix(self, entry_prefix: str | None) -> None:
    if self.is_alternating:
      return
    with self._lock:
      self._prefixes = [entry_prefix] if entry_prefix else []

  def get_entry_prefix(self) -> str | None:
    with self._lock:
      return self._prefixes[self._prefix_index] if self._prefixes else None

  def get_saved_count(self) -> int:
    with self._lock:
      return self._saved_count

  def get_prefix_index(self) -> int:
    with self._lock:
      return self._prefix_index

  def _advance_prefix(self) -> None:
    """Toggle prefix index if alternating. Must be called under lock."""
    if self.is_alternating:
      self._prefix_index = 1 - self._prefix_index

  def discard(self) -> None:
    with self._lock:
      prefix = self._prefixes[self._prefix_index] if self._prefixes else None
      discard_prefix = f"discarded_{prefix}" if prefix else "discarded"
      self._episode_client.save(entry_prefix=discard_prefix)
      self._discarded_count += 1
      log.info("Discarded episodes: {}", self._discarded_count)
      self._advance_prefix()

  def get_discarded_count(self) -> int:
    with self._lock:
      return self._discarded_count

  def get_state(self) -> rpc_api.EpisodeObserverStateResponse:
    with self._lock:
      return self._episode_client.get_state()

  def toggle_start_stop(self) -> None:
    with self._lock:
      state = self._episode_client.get_state()
    if state.pending_save_decision:
      return
    try:
      if state.is_recording:
        self.stop()
      else:
        self.start()
    except Exception:
      log.exception("toggle_start_stop failed")
      _set_toast("Start/stop failed. Check logs.", duration_s=5.0)


@dataclasses.dataclass
class ToastState:
  message: str = ""
  expires_at: float = 0.0


_TOAST_STATE = ToastState()
_TOAST_LOCK = threading.Lock()
_RECOVERY_ALERT_DURATION_SEC = 4.0


@dataclasses.dataclass
class HardwareAlertState:
  last_error_summary: str | None = None
  recovered_at_sec: float | None = None
  recovered_expires_at_sec: float = 0.0


class UiPhase(enum.Enum):
  HARDWARE_ERROR = "hardware_error"
  PENDING_SAVE = "pending_save"
  RECORDING = "recording"
  RESETTING = "resetting"
  READY = "ready"


_HARDWARE_ALERT_STATE = HardwareAlertState()
_HARDWARE_ALERT_LOCK = threading.Lock()


def _set_toast(message: str, duration_s: float = 1.4) -> None:
  if not message:
    return
  with _TOAST_LOCK:
    _TOAST_STATE.message = message
    _TOAST_STATE.expires_at = time.time() + duration_s


def _get_toast(now: float) -> tuple[str, str]:
  with _TOAST_LOCK:
    if _TOAST_STATE.message and now < _TOAST_STATE.expires_at:
      return _TOAST_STATE.message, "toast show"
    _TOAST_STATE.message = ""
    _TOAST_STATE.expires_at = 0.0
    return "", "toast"


def _split_hardware_summary(summary: str) -> list[str]:
  return [entry.strip() for entry in summary.split(";") if entry.strip()]


def _extract_hardware_error_summary(
    state: rpc_api.EpisodeObserverStateResponse,
) -> str | None:
  hardware_healthy = getattr(state, "hardware_healthy", True)
  if hardware_healthy:
    return None

  hardware_summary = (getattr(state, "hardware_summary", "") or "").strip()
  if hardware_summary:
    return hardware_summary

  control_message = (state.control_message or "").strip()
  if control_message:
    return control_message
  return "Hardware is unhealthy."


def _derive_ui_phase(
    state: rpc_api.EpisodeObserverStateResponse,
    reset_ready_for_start: bool,
) -> UiPhase:
  if _extract_hardware_error_summary(state) is not None:
    return UiPhase.HARDWARE_ERROR
  if state.pending_save_decision:
    return UiPhase.PENDING_SAVE
  if state.is_recording:
    return UiPhase.RECORDING
  if not reset_ready_for_start:
    return UiPhase.RESETTING
  return UiPhase.READY


def _build_error_alert_content(summary: str) -> list[object]:
  issue_lines = _split_hardware_summary(summary)
  if not issue_lines:
    issue_lines = [summary]
  details: list[object] = [
      html.Div("Hardware error — recording blocked", className="alert-title"),
      html.Div(
          "Fix hardware before recording resumes.",
          className="alert-description",
      ),
  ]
  details.append(
      html.Ul(
          [html.Li(issue_line) for issue_line in issue_lines],
          className="alert-list",
      )
  )
  return details


def _build_recovered_alert_content(recovered_at_sec: float) -> list[object]:
  recovered_time = dt.datetime.fromtimestamp(recovered_at_sec).strftime(
      "%H:%M:%S"
  )
  return [
      html.Div(
          "Hardware recovered — recording unblocked", className="alert-title"
      ),
      html.Div(
          f"Recovered at {recovered_time}.",
          className="alert-description",
      ),
  ]


def _get_hardware_alert(
    summary: str | None,
    now_sec: float,
) -> tuple[list[object] | str, str]:
  with _HARDWARE_ALERT_LOCK:
    if summary is not None:
      _HARDWARE_ALERT_STATE.last_error_summary = summary
      _HARDWARE_ALERT_STATE.recovered_at_sec = None
      _HARDWARE_ALERT_STATE.recovered_expires_at_sec = 0.0
      return _build_error_alert_content(summary), "alert alert-error"

    if _HARDWARE_ALERT_STATE.last_error_summary is not None:
      _HARDWARE_ALERT_STATE.last_error_summary = None
      _HARDWARE_ALERT_STATE.recovered_at_sec = now_sec
      _HARDWARE_ALERT_STATE.recovered_expires_at_sec = (
          now_sec + _RECOVERY_ALERT_DURATION_SEC
      )

    if (
        _HARDWARE_ALERT_STATE.recovered_at_sec is not None
        and now_sec < _HARDWARE_ALERT_STATE.recovered_expires_at_sec
    ):
      return (
          _build_recovered_alert_content(
              _HARDWARE_ALERT_STATE.recovered_at_sec
          ),
          "alert alert-recovered",
      )

    _HARDWARE_ALERT_STATE.recovered_at_sec = None
    _HARDWARE_ALERT_STATE.recovered_expires_at_sec = 0.0
    return "", "alert alert-hidden"


def _build_pedal_listener(
    controller: EpisodeController,
    device_path: str,
) -> PedalListener | None:
  if evdev is None:
    raise RuntimeError("evdev is required when --enable_pedal is set.")

  def on_pedal_event(event: Event) -> None:
    if event.state != ButtonState.PRESSED:
      return
    if event.button == Button.A:
      controller.toggle_start_stop()
      return
    if event.button == Button.B:
      state = controller.get_state()
      if state.pending_save_decision:
        if controller.get_entry_prefix():
          controller.save()
          _set_toast("Episode saved.")
        else:
          _set_toast("Entry prefix required.")
      return
    if event.button == Button.C:
      state = controller.get_state()
      if state.pending_save_decision:
        controller.discard()
        _set_toast("Episode saved as discarded.")
      return

  try:
    pedal_listener = PedalListener(
        device_path=device_path,
        on_click=on_pedal_event,
    )
    pedal_listener.start()
    log.info("Pedal listener started.")
  except ValueError as exc:
    raise RuntimeError(f"Failed to start pedal listener: {exc}") from exc
  return pedal_listener


def _format_state(
    state: rpc_api.EpisodeObserverStateResponse,
) -> list[html.Div]:
  fps_text = "--" if state.fps is None else f"{state.fps:.1f}"
  hardware_healthy = getattr(state, "hardware_healthy", True)
  hardware_summary = (getattr(state, "hardware_summary", "") or "").strip()
  if hardware_healthy:
    hardware_text = "OK"
  elif hardware_summary:
    issue_count = len(_split_hardware_summary(hardware_summary))
    issue_label = "issue" if issue_count == 1 else "issues"
    hardware_text = f"ERROR ({issue_count} {issue_label})"
  else:
    hardware_text = "ERROR"
  if state.is_recording:
    recording_value = html.Div(
        "Recording", className="state-value recording-active"
    )
  else:
    recording_value = html.Div("Idle", className="state-value recording-idle")

  items = [
      ("Available", "Yes" if state.is_available else "No"),
      ("Pending Save", "Yes" if state.pending_save_decision else "No"),
      ("FPS", fps_text),
      ("Hardware", hardware_text),
      ("Task", state.task_description or "--"),
  ]
  return [
      html.Div(
          [
              html.Div("Status", className="state-label"),
              recording_value,
          ],
          className="state-item",
      ),
  ] + [
      html.Div(
          [
              html.Div(label, className="state-label"),
              html.Div(value, className="state-value"),
          ],
          className="state-item",
      )
      for label, value in items
  ]


def _button(
    label: str,
    button_id: str,
    class_name: str,
    style: dict[str, str] | None = None,
) -> html.Button:
  return html.Button(
      label,
      id=button_id,
      className=f"btn {class_name}",
      n_clicks=0,
      style=style,
  )


def _visibility_style(visible: bool) -> dict[str, str]:
  return {"display": "inline-flex"} if visible else {"display": "none"}


def _camera_visibility_style(visible: bool) -> dict[str, str]:
  return {"display": "flex"} if visible else {"display": "none"}


def _camera_key(camera: rpc_api.CameraType) -> str:
  return camera.name.lower()


def _build_camera_figure(rgb: np.ndarray) -> go.Figure:
  if rgb.dtype != np.uint8:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
  fig = go.Figure(go.Image(z=rgb))
  fig.update_layout(
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      xaxis=dict(visible=False),
      yaxis=dict(visible=False),
  )
  fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
  fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
  return fig


_EMPTY_CAMERA_FIGURE = go.Figure()
_EMPTY_CAMERA_FIGURE.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)


def _build_app(
    robot: r2client.Robot,
    controller: EpisodeController,
    reset_coordinator: ResetCoordinator,
    poll_interval_ms: int,
    enable_camera_vis: bool,
) -> Dash:
  app = Dash(__name__)
  app.title = "R2 Episode Collector"

  app.index_string = """<!DOCTYPE html>
  <html>
    <head>
      {%metas%}
      <title>{%title%}</title>
      {%favicon%}
      {%css%}
      <style>
        @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap");
        :root {
          --bg: #0f172a;
          --bg-accent: #111827;
          --card: rgba(255, 255, 255, 0.08);
          --card-border: rgba(255, 255, 255, 0.12);
          --text: #f8fafc;
          --muted: #cbd5f5;
          --start: #22c55e;
          --stop: #f97316;
          --save: #38bdf8;
          --discard: #f43f5e;
          --alert-error-bg: rgba(127, 29, 29, 0.65);
          --alert-error-border: rgba(248, 113, 113, 0.95);
          --alert-error-text: #fee2e2;
          --alert-recovered-bg: rgba(20, 83, 45, 0.6);
          --alert-recovered-border: rgba(74, 222, 128, 0.9);
          --alert-recovered-text: #dcfce7;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          color: var(--text);
          background:
            radial-gradient(1200px 800px at 15% 10%, rgba(56, 189, 248, 0.18), transparent),
            radial-gradient(900px 600px at 85% 0%, rgba(244, 63, 94, 0.18), transparent),
            linear-gradient(135deg, var(--bg), var(--bg-accent));
          font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
        }
        .page {
          min-height: 100vh;
          padding: 32px 16px 48px;
          display: flex;
          justify-content: center;
        }
        .shell {
          width: min(980px, 100%);
          display: grid;
          gap: 24px;
        }
        .hero {
          padding: 28px;
          border: 1px solid var(--card-border);
          background: var(--card);
          border-radius: 20px;
          backdrop-filter: blur(12px);
        }
        .title {
          font-size: clamp(28px, 4vw, 40px);
          margin: 0 0 8px;
        }
        .subtitle {
          margin: 0;
          color: var(--muted);
          font-size: 16px;
        }
        .alert {
          margin-top: 18px;
          padding: 14px 16px;
          border-radius: 12px;
          border: 1px solid transparent;
        }
        .alert-hidden {
          display: none;
        }
        .alert-error {
          display: block;
          background: var(--alert-error-bg);
          border-color: var(--alert-error-border);
          color: var(--alert-error-text);
          box-shadow: 0 10px 24px rgba(127, 29, 29, 0.34);
        }
        .alert-recovered {
          display: block;
          background: var(--alert-recovered-bg);
          border-color: var(--alert-recovered-border);
          color: var(--alert-recovered-text);
          box-shadow: 0 10px 24px rgba(21, 128, 61, 0.25);
        }
        .alert-title {
          font-size: 16px;
          font-weight: 700;
          margin: 0;
        }
        .alert-description {
          margin-top: 6px;
          font-size: 14px;
          line-height: 1.35;
        }
        .alert-list {
          margin: 10px 0 0;
          padding-left: 18px;
          display: grid;
          gap: 4px;
          font-size: 13px;
          line-height: 1.35;
        }
        .controls {
          display: grid;
          gap: 12px;
          grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
          margin-top: 20px;
        }
        .btn {
          border: none;
          padding: 14px 18px;
          font-size: 16px;
          font-weight: 600;
          border-radius: 14px;
          cursor: pointer;
          color: #0b1120;
          transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
        }
        .btn:active {
          transform: translateY(0);
        }
        .btn:disabled {
          opacity: 0.45;
          cursor: not-allowed;
          box-shadow: none;
        }
        .btn-start { background: var(--start); }
        .btn-stop { background: var(--stop); }
        .btn-save { background: var(--save); }
        .btn-discard { background: var(--discard); color: #fff; }
        .status {
          margin-top: 18px;
          font-size: 14px;
          color: var(--muted);
          min-height: 18px;
        }
        .status-error {
          margin-top: 18px;
          font-size: 14px;
          font-weight: 600;
          color: var(--alert-error-text);
          background: var(--alert-error-bg);
          border: 1px solid var(--alert-error-border);
          border-radius: 8px;
          padding: 10px 14px;
          min-height: 18px;
        }
        .prefix-row {
          margin-top: 18px;
          display: grid;
          gap: 8px;
        }
        .prefix-label {
          text-transform: uppercase;
          font-size: 11px;
          letter-spacing: 0.14em;
          color: var(--muted);
        }
        .prefix-input {
          width: 100%;
          padding: 12px 14px;
          border-radius: 12px;
          border: 1px solid var(--card-border);
          background: rgba(15, 23, 42, 0.6);
          color: var(--text);
          font-size: 15px;
        }
        .toast {
          position: fixed;
          bottom: 24px;
          right: 24px;
          padding: 12px 16px;
          border-radius: 12px;
          background: rgba(15, 23, 42, 0.9);
          border: 1px solid var(--card-border);
          color: var(--text);
          font-weight: 600;
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
          transform: translateY(8px);
          opacity: 0;
          transition: opacity 0.2s ease, transform 0.2s ease;
          z-index: 20;
        }
        .toast.show {
          opacity: 1;
          transform: translateY(0);
        }
        .grid {
          display: grid;
          gap: 16px;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }
        .camera-controls {
          display: flex;
          flex-wrap: wrap;
          gap: 16px;
          align-items: center;
          justify-content: space-between;
          padding: 14px 18px;
          border-radius: 16px;
          border: 1px solid var(--card-border);
          background: rgba(15, 23, 42, 0.45);
        }
        .camera-controls label {
          display: block;
          font-size: 12px;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--muted);
          margin-bottom: 6px;
        }
        .camera-toggle {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          color: var(--text);
          font-size: 14px;
        }
        .camera-toggle input {
          margin-right: 6px;
        }
        .camera-row {
          display: flex;
          flex-wrap: wrap;
          gap: 16px;
          justify-content: center;
          align-items: stretch;
        }
        .camera-card {
          flex: 1 1 240px;
          max-width: 320px;
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .camera-label {
          text-transform: uppercase;
          font-size: 11px;
          letter-spacing: 0.18em;
          color: var(--muted);
        }
        .camera-graph {
          height: 200px;
        }
        .card {
          padding: 18px;
          border-radius: 18px;
          border: 1px solid var(--card-border);
          background: rgba(15, 23, 42, 0.45);
        }
        .state-item {
          display: flex;
          flex-direction: column;
          gap: 4px;
          padding: 8px 0;
        }
        .state-label {
          text-transform: uppercase;
          font-size: 11px;
          letter-spacing: 0.14em;
          color: var(--muted);
        }
        .state-value {
          font-size: 16px;
          font-weight: 600;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .recording-active {
          color: #fff;
          background: var(--stop);
          padding: 4px 12px;
          border-radius: 6px;
          display: inline-block;
          animation: recording-pulse 1.2s ease-in-out infinite;
        }
        @keyframes recording-pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
        .recording-idle {
          color: var(--muted);
          padding: 4px 12px;
          border-radius: 6px;
          display: inline-block;
        }
        @media (max-width: 640px) {
          .hero { padding: 22px; }
          .controls { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
          .camera-controls { flex-direction: column; align-items: stretch; }
          .camera-card { max-width: 100%; }
          .camera-graph { height: 180px; }
        }
      </style>
    </head>
    <body>
      {%app_entry%}
      <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
      </footer>
    </body>
  </html>"""

  app.layout = html.Div(
      className="page",
      children=[
          html.Div(
              className="shell",
              children=[
                  html.Div(
                      className="hero",
                      children=[
                          html.H1("Episode Collector", className="title"),
                          html.P(
                              "Control episode capture from your browser.",
                              className="subtitle",
                          ),
                          html.Div(
                              id="hardware-alert",
                              className="alert alert-hidden",
                          ),
                          html.Div(
                              className="controls",
                              children=[
                                  _button(
                                      "Start",
                                      "btn-start",
                                      "btn-start",
                                      style=_visibility_style(True),
                                  ),
                                  _button(
                                      "Stop",
                                      "btn-stop",
                                      "btn-stop",
                                      style=_visibility_style(False),
                                  ),
                                  _button(
                                      "Save",
                                      "btn-save",
                                      "btn-save",
                                      style=_visibility_style(False),
                                  ),
                                  _button(
                                      "Discard",
                                      "btn-discard",
                                      "btn-discard",
                                      style=_visibility_style(False),
                                  ),
                              ],
                          ),
                          html.Div(
                              className="prefix-row",
                              children=[
                                  html.Div(
                                      (
                                          "Active Prefix (1 of 2)"
                                          if controller.is_alternating
                                          else "Entry Prefix"
                                      ),
                                      id="prefix-label",
                                      className="prefix-label",
                                  ),
                                  dcc.Input(
                                      id="entry-prefix-input",
                                      className="prefix-input",
                                      type="text",
                                      value=controller.get_entry_prefix() or "",
                                      debounce=False,
                                      placeholder="leave empty to use default",
                                      disabled=controller.is_alternating,
                                  ),
                              ],
                          ),
                          html.Div(id="action-status", className="status"),
                      ],
                  ),
                  html.Div(
                      className="camera-controls",
                      style=_camera_visibility_style(enable_camera_vis),
                      children=[
                          html.Div(
                              children=[
                                  html.Label("Camera toggles"),
                                  dcc.Checklist(
                                      id="camera-toggle",
                                      className="camera-toggle",
                                      options=[
                                          {
                                              "label": "Wrist",
                                              "value": _camera_key(
                                                  rpc_api.CameraType.WRIST
                                              ),
                                          },
                                          {
                                              "label": "Scene Left",
                                              "value": _camera_key(
                                                  rpc_api.CameraType.SCENE_LEFT
                                              ),
                                          },
                                          {
                                              "label": "Scene Right",
                                              "value": _camera_key(
                                                  rpc_api.CameraType.SCENE_RIGHT
                                              ),
                                          },
                                      ],
                                      value=[
                                          _camera_key(rpc_api.CameraType.WRIST),
                                          _camera_key(
                                              rpc_api.CameraType.SCENE_LEFT
                                          ),
                                          _camera_key(
                                              rpc_api.CameraType.SCENE_RIGHT
                                          ),
                                      ],
                                      inputStyle={"margin-right": "6px"},
                                  ),
                              ],
                          ),
                          html.Div(
                              style={"minWidth": "220px", "flex": "1"},
                              children=[
                                  html.Label("Camera size"),
                                  dcc.Slider(
                                      id="camera-size",
                                      min=160,
                                      max=360,
                                      step=10,
                                      value=200,
                                      marks=None,
                                      tooltip={
                                          "placement": "bottom",
                                          "always_visible": False,
                                      },
                                  ),
                              ],
                          ),
                      ],
                  ),
                  html.Div(
                      id="camera-row",
                      className="camera-row",
                      style=_camera_visibility_style(enable_camera_vis),
                      children=[
                          html.Div(
                              id=f"camera-card-{_camera_key(camera)}",
                              className="card camera-card",
                              style=_camera_visibility_style(False),
                              children=[
                                  html.Div(label, className="camera-label"),
                                  dcc.Graph(
                                      id=f"camera-graph-{_camera_key(camera)}",
                                      className="camera-graph",
                                      figure=_EMPTY_CAMERA_FIGURE,
                                      config={"displayModeBar": False},
                                  ),
                              ],
                          )
                          for camera, label in (
                              (rpc_api.CameraType.WRIST, "Wrist"),
                              (rpc_api.CameraType.SCENE_LEFT, "Scene Left"),
                              (rpc_api.CameraType.SCENE_RIGHT, "Scene Right"),
                          )
                      ],
                  ),
                  html.Div(
                      className="grid",
                      children=[
                          html.Div(
                              className="card",
                              children=[
                                  html.H3(
                                      "Episode State", className="subtitle"
                                  ),
                                  html.Div(id="state-status"),
                              ],
                          ),
                          html.Div(
                              className="card",
                              children=[
                                  html.H3(
                                      "Control Message", className="subtitle"
                                  ),
                                  html.Div(
                                      id="control-message",
                                      className="state-value",
                                  ),
                              ],
                          ),
                          html.Div(
                              className="card",
                              children=[
                                  html.H3(
                                      "Saved Episodes", className="subtitle"
                                  ),
                                  html.Div(
                                      id="saved-count",
                                      className="state-value",
                                  ),
                              ],
                          ),
                          html.Div(
                              className="card",
                              children=[
                                  html.H3(
                                      "Discarded Episodes", className="subtitle"
                                  ),
                                  html.Div(
                                      id="discarded-count",
                                      className="state-value",
                                  ),
                              ],
                          ),
                      ],
                  ),
                  dcc.Interval(
                      id="state-poll",
                      interval=poll_interval_ms,
                      n_intervals=0,
                  ),
              ],
          ),
          html.Div(id="toast", className="toast"),
          dcc.Store(
              id="entry-prefix-store",
              data=controller.get_entry_prefix() or "",
          ),
      ],
  )

  @app.callback(
      Output("entry-prefix-store", "data"),
      Input("entry-prefix-input", "value"),
  )
  def update_entry_prefix(value):
    entry_value = (value or "").strip()
    controller.set_entry_prefix(entry_value or None)
    return entry_value

  @app.callback(
      Output("action-status", "children"),
      Output("action-status", "className"),
      Input("btn-start", "n_clicks"),
      Input("btn-stop", "n_clicks"),
      Input("btn-save", "n_clicks"),
      Input("btn-discard", "n_clicks"),
      State("entry-prefix-input", "value"),
      prevent_initial_call=True,
  )
  def handle_action(_start, _stop, _save, _discard, entry_prefix_value):
    ctx = dash.callback_context
    if not ctx.triggered:
      return no_update, no_update
    action = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
      entry_value = (entry_prefix_value or "").strip()
      controller.set_entry_prefix(entry_value or None)
      if action == "btn-start":
        current_state = controller.get_state()
        hardware_error = _extract_hardware_error_summary(current_state)
        if hardware_error is not None:
          _set_toast("Hardware unhealthy: start blocked.", duration_s=5.0)
          return (
              "Error: cannot start while hardware is unhealthy: "
              f"{hardware_error}"
          ), "status-error"
        controller.start()
        # Verify the backend actually started recording.
        state = controller.get_state()
        if not state.is_recording:
          raise RuntimeError(
              "Backend did not start recording. Check backend logs."
          )
        verb = "Started"
      elif action == "btn-stop":
        controller.stop()
        # Verify the backend actually stopped recording.
        state = controller.get_state()
        if state.is_recording:
          raise RuntimeError(
              "Backend did not stop recording. Check backend logs."
          )
        verb = "Stopped"
      elif action == "btn-save":
        if not controller.get_entry_prefix():
          _set_toast("Entry prefix required.", duration_s=5.0)
          return (
              "Error: entry_prefix is required before saving.",
              "status-error",
          )
        controller.save()
        verb = "Saved"
        _set_toast("Episode saved.")
      elif action == "btn-discard":
        controller.discard()
        verb = "Discarded"
        _set_toast("Episode saved as discarded.")
      else:
        return no_update, no_update
      stamp = dt.datetime.now().strftime("%H:%M:%S")
      return f"{verb} episode at {stamp}.", "status"
    except Exception as exc:
      log.exception("handle_action failed for {}", action)
      _set_toast(f"Action failed: {exc}", duration_s=5.0)
      return f"Error: {exc}", "status-error"

  @app.callback(
      Output("state-status", "children"),
      Output("control-message", "children"),
      Output("hardware-alert", "children"),
      Output("hardware-alert", "className"),
      Output("btn-start", "style"),
      Output("btn-start", "disabled"),
      Output("btn-start", "children"),
      Output("btn-stop", "style"),
      Output("btn-save", "style"),
      Output("btn-discard", "style"),
      Output("saved-count", "children"),
      Output("discarded-count", "children"),
      Output("toast", "children"),
      Output("toast", "className"),
      Output("prefix-label", "children"),
      Output("entry-prefix-input", "value"),
      Input("state-poll", "n_intervals"),
  )
  def refresh_state(_tick):
    now_sec = time.time()
    toast_message, toast_class = _get_toast(now_sec)
    reset_ready_for_start = reset_coordinator.is_ready_for_start()
    reset_waiting_for_start = reset_coordinator.is_waiting_for_start()
    saved_count = controller.get_saved_count()
    discarded_count = controller.get_discarded_count()
    if controller.is_alternating:
      idx = controller.get_prefix_index()
      prefix_label = f"Active Prefix ({idx + 1} of 2)"
      prefix_value = controller.get_entry_prefix() or ""
    else:
      prefix_label = no_update
      prefix_value = no_update
    try:
      state = controller.get_state()
    except Exception as exc:
      unavailable_alert = [
          html.Div(
              "Connection error — controls unavailable", className="alert-title"
          ),
          html.Div(
              f"Error: {exc}",
              className="alert-description",
          ),
      ]
      hidden = _visibility_style(False)
      return (
          [html.Div("Unavailable")],
          f"Error: {exc}",
          unavailable_alert,
          "alert alert-error",
          hidden,
          True,
          "Start",
          hidden,
          hidden,
          hidden,
          str(saved_count),
          str(discarded_count),
          toast_message,
          toast_class,
          prefix_label,
          prefix_value,
      )
    hardware_error = _extract_hardware_error_summary(state)
    alert_children, alert_class_name = _get_hardware_alert(
        hardware_error,
        now_sec,
    )
    phase = _derive_ui_phase(
        state=state,
        reset_ready_for_start=reset_ready_for_start,
    )

    hidden = _visibility_style(False)
    start_style = hidden
    stop_style = hidden
    save_style = hidden
    discard_style = hidden
    start_disabled = True
    start_label = "Start"

    if phase == UiPhase.PENDING_SAVE:
      save_style = _visibility_style(True)
      discard_style = _visibility_style(True)
    elif phase == UiPhase.RECORDING:
      stop_style = _visibility_style(True)
    elif phase == UiPhase.RESETTING:
      start_style = _visibility_style(True)
      start_disabled = True
      start_label = "Resetting..."
    elif phase == UiPhase.READY:
      start_style = _visibility_style(True)
      start_disabled = False

    control_message = state.control_message or "--"
    if phase == UiPhase.HARDWARE_ERROR and hardware_error is not None:
      control_message = f"Error: {hardware_error}"
    elif phase == UiPhase.RESETTING:
      if control_message == "--":
        control_message = "Waiting for start..."
      else:
        control_message = f"Waiting for start... {control_message}"
    elif phase == UiPhase.READY and reset_waiting_for_start:
      if control_message == "--":
        control_message = "Ready for start."

    return (
        _format_state(state),
        control_message,
        alert_children,
        alert_class_name,
        start_style,
        start_disabled,
        start_label,
        stop_style,
        save_style,
        discard_style,
        str(saved_count),
        str(discarded_count),
        toast_message,
        toast_class,
        prefix_label,
        prefix_value,
    )

  camera_specs = (
      (rpc_api.CameraType.WRIST, _camera_key(rpc_api.CameraType.WRIST)),
      (
          rpc_api.CameraType.SCENE_LEFT,
          _camera_key(rpc_api.CameraType.SCENE_LEFT),
      ),
      (
          rpc_api.CameraType.SCENE_RIGHT,
          _camera_key(rpc_api.CameraType.SCENE_RIGHT),
      ),
  )

  camera_outputs = [Output("camera-row", "style")]
  for _, camera_key in camera_specs:
    camera_outputs.append(Output(f"camera-graph-{camera_key}", "figure"))
    camera_outputs.append(Output(f"camera-card-{camera_key}", "style"))
    camera_outputs.append(Output(f"camera-graph-{camera_key}", "style"))

  @app.callback(
      camera_outputs,
      Input("state-poll", "n_intervals"),
      Input("camera-toggle", "value"),
      Input("camera-size", "value"),
  )
  def refresh_cameras(_tick, enabled_cameras, camera_size):
    if camera_size is None:
      camera_size = 200
    if not enable_camera_vis:
      hidden_row = _camera_visibility_style(False)
      figures = [_EMPTY_CAMERA_FIGURE for _ in camera_specs]
      styles = [_camera_visibility_style(False) for _ in camera_specs]
      graph_styles = [{"height": f"{camera_size}px"} for _ in camera_specs]
      return tuple(
          [hidden_row]
          + [
              value
              for pair in zip(figures, styles, graph_styles)
              for value in pair
          ]
      )

    enabled = set(enabled_cameras or [])
    row_visible = _camera_visibility_style(bool(enabled))
    figures: list[go.Figure] = []
    styles: list[dict[str, str]] = []
    graph_styles: list[dict[str, str]] = []
    for camera_type, camera_key in camera_specs:
      if camera_key not in enabled:
        figures.append(_EMPTY_CAMERA_FIGURE)
        styles.append(_camera_visibility_style(False))
        graph_styles.append({"height": f"{camera_size}px"})
        continue
      try:
        camera_data = robot.raw_robot.get_camera_data(camera=camera_type)
      except Exception as ex:
        log.warning("Failed to get camera data for {}: {}", camera_type, ex)
        camera_data = None
      if camera_data is None or camera_data.rgb is None:
        figures.append(_EMPTY_CAMERA_FIGURE)
        styles.append(_camera_visibility_style(False))
        graph_styles.append({"height": f"{camera_size}px"})
        continue
      figures.append(_build_camera_figure(camera_data.rgb))
      styles.append(
          {
              **_camera_visibility_style(True),
              "maxWidth": f"{camera_size * 1.6:.0f}px",
          }
      )
      graph_styles.append({"height": f"{camera_size}px"})
    return tuple(
        [row_visible]
        + [
            value
            for pair in zip(figures, styles, graph_styles)
            for value in pair
        ]
    )

  return app


def main(argv: list[str]) -> None:
  del argv  # Unused.
  dotenv.load_dotenv()
  r2_logging.configure(service="collect-data")
  sentry.init_sentry(service="collect-data")

  robot = r2client.Robot(
      f"tcp://{FLAGS.robot_hostname}:{FLAGS.robot_port}",
      query_server_address=(
          f"tcp://{FLAGS.robot_hostname}:{FLAGS.robot_query_port}"
      ),
      training_server_address=f"tcp://localhost:{rpc_api.DEFAULT_MODEL_TRAINER_PORT}",
  )

  # If continuous teleop, then  make sure to align the leader with the follower
  # before starting the app, to avoid sudden movements when the first reset
  # begins.
  if FLAGS.continuous_teleop:
    log.info("Aligning leader and follower for continuous teleop...")
    robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)
    motion_future = robot.behaviour.align_leader_with_follower(
        timeout_seconds=2.0,
        threshold=0.1,
    )
    motion_future.result()
    log.info("Alignment complete.")

  # Parse start_trajectory - treat "None" string as None.
  start_trajectory = FLAGS.start_trajectory
  if start_trajectory and start_trajectory.lower() == "none":
    start_trajectory = None

  prefixes = FLAGS.entry_prefix or []
  if len(prefixes) > 2:
    raise absl_app.UsageError("--entry_prefix accepts at most 2 values.")

  reset_coordinator = ResetCoordinator(
      robot,
      continuous_teleop=FLAGS.continuous_teleop,
      start_trajectory=start_trajectory,
  )
  reset_coordinator.request_reset()
  controller = EpisodeController(
      robot.episode_observer,
      reset_coordinator,
      prefixes=prefixes,
  )
  pedal_listener = None
  if FLAGS.enable_pedal:
    pedal_listener = _build_pedal_listener(controller, FLAGS.pedal_device_path)

    def _cleanup(sig, frame):
      del sig, frame  # Unused.
      if pedal_listener is not None:
        pedal_listener.stop()
      reset_coordinator.stop()
      raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

  app = _build_app(
      robot,
      controller,
      reset_coordinator,
      FLAGS.poll_interval_ms,
      FLAGS.enable_camera_vis,
  )
  app.run(host=FLAGS.web_host, port=FLAGS.web_port, debug=False)

  if pedal_listener is not None:
    pedal_listener.stop()
  reset_coordinator.stop()


if __name__ == "__main__":
  try:
    absl_app.run(main)
  except SystemExit:
    raise
  except KeyboardInterrupt:
    pass
  except Exception:
    sentry.capture_exception()
    raise
