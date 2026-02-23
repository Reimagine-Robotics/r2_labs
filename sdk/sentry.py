"""Shared Sentry initialisation for r2_labs services."""

import os
import socket
from typing import Any, Literal

from loguru import logger as log

try:
  import sentry_sdk
except ImportError:
  sentry_sdk = None


def init_sentry(*, service: str) -> None:
  dsn: str = os.getenv("SENTRY_DSN", "").strip()
  if not dsn:
    log.warning("Sentry disabled: SENTRY_DSN is not set.")
    return

  if sentry_sdk is None:
    log.warning("Sentry disabled: sentry-sdk is not installed.")
    return

  traces_sample_rate: float = 0.0
  raw: str | None = os.getenv("SENTRY_TRACES_SAMPLE_RATE")
  if raw:
    try:
      traces_sample_rate = float(raw)
    except ValueError:
      log.warning(
          "Invalid SENTRY_TRACES_SAMPLE_RATE={!r}; defaulting to 0.0.",
          raw,
      )

  enable_logs: bool = os.getenv("ENABLE_SENTRY_LOGS", "").lower() in (
      "1",
      "true",
      "yes",
  )
  server_name: str = os.getenv("SENTRY_SERVER_NAME") or socket.gethostname()
  environment: str = os.getenv("SENTRY_ENVIRONMENT", "production")

  def before_send(event: dict[str, Any], hint: Any) -> dict[str, Any]:
    del hint
    event.setdefault("tags", {})["service"] = service
    return event

  try:
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        server_name=server_name,
        before_send=before_send,  # pyright: ignore[reportArgumentType]
        enable_logs=enable_logs,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    log.warning("Sentry init failed: {}", exc)


LogLevel = Literal["fatal", "critical", "error", "warning", "info", "debug"]


def capture_exception(
    exc: BaseException | None = None,
    *,
    level: LogLevel = "error",
) -> None:
  if sentry_sdk is None:
    return
  try:
    with sentry_sdk.new_scope() as scope:
      scope.set_level(level)
      scope.capture_exception(exc)
  except Exception:  # pylint: disable=broad-exception-caught
    log.warning("Failed to capture exception to Sentry.")
