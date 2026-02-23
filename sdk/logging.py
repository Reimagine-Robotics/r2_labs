"""Logging configuration for r2_labs services."""

import os
import pathlib
import sys

from loguru import logger as loguru_logger

_DEFAULT_LOG_DIR = "/var/log/r2"


def configure(*, service: str, log_level: str | None = None) -> None:
  """Configure loguru logging with stderr and optional rotating file handler.

  Args:
    service: Service name used for the log file name.
    log_level: Override log level. Defaults to LOGURU_LEVEL env var or INFO.
  """
  if log_level is None:
    log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()

  # Reset loguru and add coloured stderr handler.
  loguru_logger.remove()
  loguru_logger.add(sys.stderr, level=log_level, colorize=True)

  # Rotating file handler.
  log_dir = os.environ.get("R2_LOG_DIR")
  if log_dir is None:
    log_dir = _DEFAULT_LOG_DIR
  else:
    log_dir = log_dir.strip()

  file_handler_active = False
  if log_dir:
    file_handler_active = _add_rotating_file_handler(
        service, log_dir, log_level
    )

  _log_configuration_summary(service, log_dir, log_level, file_handler_active)


def _add_rotating_file_handler(
    service: str,
    log_dir: str,
    log_level: str,
) -> bool:
  """Adds a rotating file handler for the given service."""
  log_path = pathlib.Path(log_dir)

  try:
    log_path.mkdir(parents=True, exist_ok=True)
  except OSError as e:
    loguru_logger.warning(
        "file logging disabled: cannot create log dir {}: {}",
        log_dir,
        e,
    )
    return False

  log_file = log_path / f"{service}.log"
  if log_file.exists() and not os.access(log_file, os.W_OK):
    loguru_logger.warning(
        "file logging disabled: no write permission on {}",
        log_file,
    )
    return False
  if not os.access(log_path, os.W_OK | os.X_OK):
    loguru_logger.warning(
        "file logging disabled: no write permission on {}",
        log_path,
    )
    return False

  loguru_logger.add(
      log_file,
      level=log_level,
      rotation="50 MB",
      retention=10,
      encoding="utf-8",
  )
  return True


def _log_configuration_summary(
    service: str,
    log_dir: str,
    log_level: str,
    file_handler_active: bool,
) -> None:
  """Logs a summary of active logging sinks."""
  sentry_dsn = os.environ.get("SENTRY_DSN", "").strip()
  sentry_enabled = bool(sentry_dsn)
  sentry_logs_enabled = sentry_enabled and os.environ.get(
      "ENABLE_SENTRY_LOGS", ""
  ).lower() in ("1", "true", "yes")

  file_handler_status: str | bool = False
  if file_handler_active:
    file_handler_status = str(pathlib.Path(log_dir) / f"{service}.log")

  loguru_logger.info(
      "logging configured | service={} level={}"
      " sentry={} sentry_logs={} file_handler={}",
      service,
      log_level,
      sentry_enabled,
      sentry_logs_enabled,
      file_handler_status,
  )
