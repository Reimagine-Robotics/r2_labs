"""Tests for version resolution and the version-mismatch message."""

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
