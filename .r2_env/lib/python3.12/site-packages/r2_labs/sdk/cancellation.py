"""Process-wide emergency cancellation of in-flight robot behaviours.

Signals reach only the main thread while behaviours run in worker threads, so a
main-thread SIGINT handler reaches into a process-wide registry and fires each
in-flight behaviour's cancel callback (which stops the robot via a server RPC).
"""

import atexit
import signal
import threading
from typing import Any, Callable

from loguru import logger as log

# Each entry is one in-flight behaviour's cancel callback, keyed by an opaque
# token so it can be removed when the behaviour resolves. A WeakValueDictionary
# is unsuitable: the callbacks are closures with no other strong reference, so
# we hold them strongly and rely on explicit unregister in a finally.
_callbacks: dict[int, Callable[[], None]] = {}
_lock: threading.Lock = threading.Lock()

_install_lock: threading.Lock = threading.Lock()
_handler_installed: bool = False
_previous_sigint_handler: Any = None
_atexit_registered: bool = False


def register_cancel(callback: Callable[[], None]) -> int:
  """Register an in-flight behaviour's cancel callback; returns a token."""
  token = id(callback)
  with _lock:
    _callbacks[token] = callback
  return token


def unregister_cancel(token: int) -> None:
  """Remove a previously registered cancel callback by its token."""
  with _lock:
    _callbacks.pop(token, None)


def cancel_all_inflight() -> int:
  """Fire every in-flight cancel callback (best-effort). Returns the count.

  Snapshots the callbacks under the lock then releases it before invoking them
  (each issues a blocking RPC); per-callback errors are logged and swallowed so
  one failure cannot block the rest. This is the seam the SIGINT handler and the
  atexit hook call, and that tests call directly.
  """
  with _lock:
    callbacks = list(_callbacks.values())
  for cancel in callbacks:
    try:
      cancel()
    except Exception as exc:  # pylint: disable=broad-except
      log.warning("emergency cancel failed | cause={}", exc)
  return len(callbacks)


def _handle_sigint(signum: int, frame: Any) -> None:
  """Main-thread SIGINT handler: cancel everything, then defer to the prior."""
  log.warning("SIGINT received | cancelling in-flight behaviours")
  cancel_all_inflight()
  prev = _previous_sigint_handler
  if callable(prev) and prev is not signal.default_int_handler:
    prev(signum, frame)  # honour a user-installed handler
  elif prev == signal.SIG_IGN:
    return
  else:
    raise KeyboardInterrupt  # SIG_DFL / default: let the script stop


def install_handlers() -> None:
  """Install the SIGINT handler and atexit sweep exactly once (idempotent).

  No-op (with a debug log) if not on the main thread, since signal.signal may
  only be called from the main thread.
  """
  global _handler_installed, _previous_sigint_handler, _atexit_registered
  with _install_lock:
    if _handler_installed:
      return
    if threading.current_thread() is not threading.main_thread():
      log.debug("not on main thread | skipping SIGINT handler install")
      return
    _previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)
    if not _atexit_registered:
      atexit.register(cancel_all_inflight)
      _atexit_registered = True
    _handler_installed = True
