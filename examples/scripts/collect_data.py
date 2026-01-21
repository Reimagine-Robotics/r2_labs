"""Dash UI for episode collection using the r2_labs SDK.

Example to run:

uv run python r2_labs/examples/scripts/collect_data.py \
  --robot_hostname=akhilraju-home.local \
  --enable_pedal \
  --entry_prefix=test_rectify_open_latch

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
import logging
import signal
import threading
import time

import dash
from dash import Dash, Input, Output, dcc, html, no_update

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
    "127.0.0.1",
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
flags.DEFINE_string(
    "entry_prefix",
    None,
    "Optional entry prefix for saved episodes.",
)


def episode_reset(
    robot: r2client.Robot,
    start_event: threading.Event,
    ready_event: threading.Event,
    waiting_event: threading.Event,
    ready_for_start_event: threading.Event,
) -> None:
  """Called at the beginning of each episode to reset things.

  You can update this function to put in your own reset logic.
  """

  # Add your reset logic here.

  # For example, moving to a saved pose.
  robot.exec_mode.set_execution_mode(rpc_api.ExecutionMode.READY)
  motion_future = robot.behaviour.trajectory_motion(
      trajectory_name="Pre-insert motion Rectify",
      motion_type=rpc_api.TrajectoryMotionType.GO_TO_END,
      static_gripper=False,
      period_seconds=None,
  )
  print("Moving to reset pose ...")
  motion_future.result()

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


class PedalListener:
  """Listen to a 3-button foot pedal via evdev."""

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

  def _event_loop(self):
    assert self._device is not None, "Device must be opened first"
    for event in self._device.read_loop():
      if not self._running:
        break
      if Event.is_pedal_event(event):
        pedal_event = Event.from_evdev_event(event)
        self._on_click(pedal_event)

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
    if self._thread is not None:
      self._thread.join()
      self._thread = None
    if self._device is not None:
      self._device.close()


class ResetCoordinator:
  """Coordinate episode resets and start triggers."""

  def __init__(self, robot: r2client.Robot) -> None:
    self._robot = robot
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
      episode_reset(
          self._robot,
          self._start_requested,
          self._ready,
          self._waiting_for_start,
          self._ready_for_start,
      )

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
      entry_prefix: str | None,
  ) -> None:
    self._episode_client = episode_client
    self._reset_coordinator = reset_coordinator
    self._entry_prefix = entry_prefix
    self._lock = threading.Lock()
    self._saved_count = 0

  def start(self) -> None:
    if not self._reset_coordinator.request_start():
      return
    self._reset_coordinator.wait_until_ready()
    with self._lock:
      self._episode_client.start()

  def stop(self) -> None:
    with self._lock:
      self._episode_client.stop()
    self._reset_coordinator.request_reset()

  def save(self) -> None:
    with self._lock:
      self._episode_client.save(entry_prefix=self._entry_prefix)
      self._saved_count += 1
      logging.info("Saved episodes: %d", self._saved_count)

  def set_entry_prefix(self, entry_prefix: str | None) -> None:
    with self._lock:
      self._entry_prefix = entry_prefix

  def get_saved_count(self) -> int:
    with self._lock:
      return self._saved_count

  def discard(self) -> None:
    with self._lock:
      self._episode_client.discard()

  def get_state(self) -> rpc_api.EpisodeObserverStateResponse:
    with self._lock:
      return self._episode_client.get_state()

  def toggle_start_stop(self) -> None:
    with self._lock:
      state = self._episode_client.get_state()
    if state.pending_save_decision:
      return
    if state.is_recording:
      self.stop()
    else:
      if not self._reset_coordinator.is_ready_for_start():
        return
      self.start()


@dataclasses.dataclass
class ToastState:
  message: str = ""
  expires_at: float = 0.0


_TOAST_STATE = ToastState()
_TOAST_LOCK = threading.Lock()


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
        controller.save()
        _set_toast("Episode saved.")
      return
    if event.button == Button.C:
      state = controller.get_state()
      if state.pending_save_decision:
        controller.discard()
        _set_toast("Episode discarded.")
      return

  try:
    pedal_listener = PedalListener(
        device_path=device_path,
        on_click=on_pedal_event,
    )
    pedal_listener.start()
    logging.info("Pedal listener started.")
  except ValueError as exc:
    logging.warning("Failed to start pedal listener: %s", exc)
    pedal_listener = None
  return pedal_listener


def _format_state(
    state: rpc_api.EpisodeObserverStateResponse,
) -> list[html.Div]:
  fps_text = "--" if state.fps is None else f"{state.fps:.1f}"
  items = [
      ("Available", "Yes" if state.is_available else "No"),
      ("Recording", "Yes" if state.is_recording else "No"),
      ("Pending Save", "Yes" if state.pending_save_decision else "No"),
      ("FPS", fps_text),
      ("Task", state.task_description or "--"),
  ]
  return [
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


def _build_app(
    controller: EpisodeController,
    reset_coordinator: ResetCoordinator,
    entry_prefix: str | None,
    poll_interval_ms: int,
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
        }
        @media (max-width: 640px) {
          .hero { padding: 22px; }
          .controls { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
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
                                      "Entry Prefix",
                                      className="prefix-label",
                                  ),
                                  dcc.Input(
                                      id="entry-prefix-input",
                                      className="prefix-input",
                                      type="text",
                                      value=entry_prefix or "",
                                      debounce=True,
                                      placeholder="leave empty to use default",
                                  ),
                              ],
                          ),
                          html.Div(id="action-status", className="status"),
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
              data=entry_prefix or "",
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
      Input("btn-start", "n_clicks"),
      Input("btn-stop", "n_clicks"),
      Input("btn-save", "n_clicks"),
      Input("btn-discard", "n_clicks"),
      prevent_initial_call=True,
  )
  def handle_action(_start, _stop, _save, _discard):
    ctx = dash.callback_context
    if not ctx.triggered:
      return no_update
    action = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
      if action == "btn-start":
        controller.start()
        verb = "Started"
      elif action == "btn-stop":
        controller.stop()
        verb = "Stopped"
      elif action == "btn-save":
        controller.save()
        verb = "Saved"
        _set_toast("Episode saved.")
      elif action == "btn-discard":
        controller.discard()
        verb = "Discarded"
        _set_toast("Episode discarded.")
      else:
        return no_update
      stamp = dt.datetime.now().strftime("%H:%M:%S")
      return f"{verb} episode at {stamp}."
    except Exception as exc:
      _set_toast("Action failed.")
      return f"Error: {exc}"

  @app.callback(
      Output("state-status", "children"),
      Output("control-message", "children"),
      Output("btn-start", "style"),
      Output("btn-start", "disabled"),
      Output("btn-start", "children"),
      Output("btn-stop", "style"),
      Output("btn-save", "style"),
      Output("btn-discard", "style"),
      Output("saved-count", "children"),
      Output("toast", "children"),
      Output("toast", "className"),
      Input("state-poll", "n_intervals"),
  )
  def refresh_state(_tick):
    toast_message, toast_class = _get_toast(time.time())
    start_disabled = not reset_coordinator.is_ready_for_start()
    start_label = "Resetting..." if start_disabled else "Start"
    saved_count = controller.get_saved_count()
    try:
      state = controller.get_state()
    except Exception as exc:
      start_style = _visibility_style(True)
      hidden = _visibility_style(False)
      return (
          [html.Div("Unavailable")],
          f"Error: {exc}",
          start_style,
          True,
          "Resetting...",
          hidden,
          hidden,
          hidden,
          str(saved_count),
          toast_message,
          toast_class,
      )
    if state.pending_save_decision:
      start_style = _visibility_style(False)
      stop_style = _visibility_style(False)
      save_style = _visibility_style(True)
      discard_style = _visibility_style(True)
    elif state.is_recording:
      start_style = _visibility_style(False)
      stop_style = _visibility_style(True)
      save_style = _visibility_style(False)
      discard_style = _visibility_style(False)
    else:
      start_style = _visibility_style(True)
      stop_style = _visibility_style(False)
      save_style = _visibility_style(False)
      discard_style = _visibility_style(False)
    control_message = state.control_message or "--"
    if reset_coordinator.is_waiting_for_start():
      if control_message == "--":
        control_message = "Waiting for start..."
      else:
        control_message = f"Waiting for start... {control_message}"
    return (
        _format_state(state),
        control_message,
        start_style,
        start_disabled,
        start_label,
        stop_style,
        save_style,
        discard_style,
        str(saved_count),
        toast_message,
        toast_class,
    )

  return app


def main(argv: list[str]) -> None:
  del argv  # Unused.
  logging.basicConfig(level=logging.INFO)

  robot = r2client.Robot(
      f"tcp://{FLAGS.robot_hostname}:{FLAGS.robot_port}",
      query_server_address=(
          f"tcp://{FLAGS.robot_hostname}:{FLAGS.robot_query_port}"
      ),
  )

  reset_coordinator = ResetCoordinator(robot)
  reset_coordinator.request_reset()
  controller = EpisodeController(
      robot.episode_observer,
      reset_coordinator,
      FLAGS.entry_prefix,
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
      controller,
      reset_coordinator,
      FLAGS.entry_prefix,
      FLAGS.poll_interval_ms,
  )
  app.run(host=FLAGS.web_host, port=FLAGS.web_port, debug=False)

  if pedal_listener is not None:
    pedal_listener.stop()
  reset_coordinator.stop()


if __name__ == "__main__":
  absl_app.run(main)
