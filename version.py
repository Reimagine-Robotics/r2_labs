"""Version resolution and the SDK/backend version-mismatch check.

Kept as a leaf module (stdlib imports only) so the RPC server can import it
without pulling the whole ``r2_labs`` package init, and so the comparison stays
pure and unit-testable. This is the Python counterpart of the extension's
``versionCheck.ts``; the two are parallel ports (TS vs Python), not a shared
module.

The check is client-side and advisory: the backend advertises its raw version
(never a compatibility verdict, which a stale backend would freeze), and the
client decides. Revisit server-advertised policy if artifacts ever version
independently or need a real compat range.
"""

import functools
import importlib.metadata
import pathlib
import tomllib

_PYPROJECT = pathlib.Path(__file__).parent / "pyproject.toml"


@functools.cache
def get_version() -> str | None:
  """Resolved distribution version, or None if genuinely undeterminable.

  A pip-installed client wheel resolves via importlib.metadata. The backend
  runs r2_labs from the monorepo checkout (not installed under its own dist
  name), so it falls back to reading the checked-out pyproject.toml — which the
  release stamps to the release version. None when neither is available; callers
  that compare versions treat None as "unknown, don't compare".

  Cached: the version is fixed for the process's lifetime, so callers on hot
  paths (every RPC ping reply, every /config response) don't repeat the metadata
  lookup / pyproject read.
  """
  try:
    return importlib.metadata.version("r2-labs")
  except importlib.metadata.PackageNotFoundError:
    pass
  try:
    return tomllib.loads(_PYPROJECT.read_text())["project"]["version"]
  except (OSError, KeyError, tomllib.TOMLDecodeError):
    return None


def _major_minor(version: str) -> str | None:
  """`major.minor` of a clean release version, else None (not comparable).

  Patch and prerelease segments are ignored (0.2.0, 0.2.3, 0.2.0rc1 -> "0.2").
  A `+local` build segment means "not a clean published release" (PyPI forbids
  local versions), so such builds are treated as not comparable.
  """
  if "+" in version:
    return None
  parts = version.split(".")
  if len(parts) < 2:
    return None
  major, minor = parts[0], parts[1]
  if not (major.isdigit() and minor.isdigit()):
    return None
  return f"{major}.{minor}"


def version_mismatch_message(
    own_version: str | None, backend_version: str | None
) -> str | None:
  """Warning message when the SDK and backend differ, else None.

  Compares on major.minor (patch/prerelease tolerated). Returns None — i.e.
  "don't warn" — when either version is unknown or not a clean release. Unlike
  the extension there is no dev-mode gate: the SDK and backend read the same
  r2_labs version source, so a dev running both from one checkout matches.
  """
  own = _major_minor(own_version) if own_version else None
  backend = _major_minor(backend_version) if backend_version else None
  if own is None or backend is None or own == backend:
    return None
  return (
      f"r2_labs SDK version ({own_version}) doesn't match the robot backend "
      f"({backend_version}). Update the SDK or the backend so they're on the "
      "same release, or you may hit unexpected behaviour."
  )
