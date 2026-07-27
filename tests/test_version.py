"""Tests for version resolution and the version-mismatch message."""

import pytest

from r2_labs import version


class TestVersionMismatchMessage:

  def test_warns_on_minor_mismatch(self):
    message = version.version_mismatch_message("0.2.0", "0.3.0")
    assert message is not None
    assert "0.2.0" in message and "0.3.0" in message

  def test_warns_on_major_mismatch(self):
    assert version.version_mismatch_message("1.2.0", "2.2.0") is not None

  def test_no_warning_when_versions_match(self):
    assert version.version_mismatch_message("0.2.0", "0.2.0") is None

  def test_tolerates_patch_differences(self):
    assert version.version_mismatch_message("0.2.0", "0.2.3") is None

  def test_tolerates_prerelease_within_same_major_minor(self):
    assert version.version_mismatch_message("0.2.0", "0.2.0rc1") is None

  def test_skips_local_build_versions(self):
    # +local means "not a clean published release" -> skip, not warn.
    assert version.version_mismatch_message("0.2.0", "0.3.0+dirty") is None

  def test_no_warning_when_backend_version_unknown(self):
    assert version.version_mismatch_message("0.2.0", None) is None

  def test_no_warning_when_own_version_unknown(self):
    assert version.version_mismatch_message(None, "0.3.0") is None

  def test_no_warning_when_unparseable(self):
    assert version.version_mismatch_message("0.2.0", "garbage") is None


class TestGetVersion:

  def test_returns_a_version_in_this_checkout(self):
    # Resolvable here (installed metadata or the checked-out pyproject).
    resolved = version.get_version()
    assert resolved is not None
    assert version._major_minor(resolved) is not None


class TestGetCommit:

  @pytest.fixture(autouse=True)
  def _clear_cache(self):
    # get_commit is @functools.cache; clear around each test so env changes
    # take effect and don't leak into other tests.
    version.get_commit.cache_clear()
    yield
    version.get_commit.cache_clear()

  def test_reads_git_commit_sha(self, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123def456")
    assert version.get_commit() == "abc123def456"

  def test_keeps_dirty_suffix(self, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123def456-dirty")
    assert version.get_commit() == "abc123def456-dirty"

  def test_strips_surrounding_whitespace(self, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "  abc123def456  ")
    assert version.get_commit() == "abc123def456"

  def test_none_on_unknown_sentinel(self, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "unknown")
    assert version.get_commit() is None

  def test_none_when_unset(self, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)
    assert version.get_commit() is None
