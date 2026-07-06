"""Logging configuration for r2_labs services.

Logs go to stderr only. Under the docker stacks that stream is captured
by the json-file driver and shipped to the site's Loki by the on-prem
monitoring stack; the tmux-era rotating file handler (R2_LOG_DIR,
default /var/log/r2) was removed with the tmux stack.
"""

import os
import sys

from loguru import logger as loguru_logger


def configure(*, service: str, log_level: str | None = None) -> None:
  """Configure loguru logging to stderr.

  Args:
    service: Service name, reported in the configuration summary.
    log_level: Override log level. Defaults to LOGURU_LEVEL env var or INFO.
  """
  if log_level is None:
    log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()

  # Reset loguru and add coloured stderr handler.
  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=log_level, colorize=True)

  _log_configuration_summary(service, log_level)


def _log_configuration_summary(service: str, log_level: str) -> None:
  """Logs a summary of active logging sinks."""
  sentry_dsn = os.environ.get("SENTRY_DSN", "").strip()
  sentry_enabled = bool(sentry_dsn)
  sentry_logs_enabled = sentry_enabled and os.environ.get(
      "ENABLE_SENTRY_LOGS", ""
  ).lower() in ("1", "true", "yes")

  loguru_logger.info(
      "logging configured | service={} level={} sentry={} sentry_logs={}",
      service,
      log_level,
      sentry_enabled,
      sentry_logs_enabled,
  )
