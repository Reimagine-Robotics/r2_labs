"""Tests for MCP response serialization."""

import dataclasses

from r2_labs.mcp_server import serialization


def test_bytes_summarized_not_dumped_as_repr():
  # Opaque binary must summarize, not fall through to str(obj) (a bytes repr).
  assert serialization.serialize(b"\xff\xd8\xff\xe0") == "<4 bytes>"
  assert serialization.serialize(b"") == "<0 bytes>"


def test_bytes_field_in_dataclass_is_summarized():
  @dataclasses.dataclass
  class _Entry:
    name: str
    preview_rgb: bytes

  assert serialization.serialize(_Entry(name="pose", preview_rgb=b"abcd")) == {
      "name": "pose",
      "preview_rgb": "<4 bytes>",
  }
