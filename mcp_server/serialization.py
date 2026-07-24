"""Recursive serialization of SDK types to JSON-safe values."""

import base64
import dataclasses
import enum
import io

import numpy as np

_SMALL_ARRAY_THRESHOLD = 100


def serialize(obj: object) -> object:
  """Convert SDK response objects into JSON-serializable values.

  Handles dataclasses, enums, numpy arrays, and standard Python types
  recursively. Large numpy arrays are summarized rather than expanded.
  """
  if obj is None or isinstance(obj, (str, int, float, bool)):
    return obj
  if isinstance(obj, bytes):
    # Opaque binary (e.g. compressed preview blobs); summarize rather than dump
    # a base64 / repr string that only bloats the agent-facing payload.
    return f"<{len(obj)} bytes>"
  if isinstance(obj, enum.Enum):
    return obj.name
  if isinstance(obj, np.ndarray):
    return _serialize_array(obj)
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return {
        field.name: serialize(getattr(obj, field.name))
        for field in dataclasses.fields(obj)
    }
  if isinstance(obj, dict):
    return {str(k): serialize(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [serialize(item) for item in obj]
  return str(obj)


def _serialize_array(arr: np.ndarray) -> object:
  """Serialize a numpy array: small arrays become lists, large get summarized."""
  if arr.size <= _SMALL_ARRAY_THRESHOLD:
    return arr.tolist()
  return {
      "shape": list(arr.shape),
      "dtype": str(arr.dtype),
      "min": float(np.min(arr)),
      "max": float(np.max(arr)),
      "mean": float(np.mean(arr)),
  }


def encode_image(arr: np.ndarray, quality: int = 85) -> str:
  """Encode an RGB uint8 image array as a base64 JPEG string."""
  if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
    raise ValueError(
        f"expected RGB uint8 array (H, W, 3), got shape={arr.shape} dtype={arr.dtype}"
    )
  from PIL import Image

  img = Image.fromarray(arr)
  buf = io.BytesIO()
  img.save(buf, format="JPEG", quality=quality)
  return base64.b64encode(buf.getvalue()).decode("ascii")
