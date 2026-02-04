#!/usr/bin/env python3
"""Launch the R2 Training Studio UI.

This is a beautiful, Apple-style web interface for configuring and monitoring
robot skill training. No notebooks required - just pure app-like experience.

Usage:
    python launch_training_ui.py [--port PORT]

Then open http://localhost:<PORT> in your browser (opens automatically).
"""

import argparse

from r2_labs.sdk.training_ui import launch_ui

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Launch R2 Training Studio UI")
  parser.add_argument(
      "--port",
      "-p",
      type=int,
      default=8000,
      help="Port to run the server on (default: 8000)",
  )
  args = parser.parse_args()

  print("\n🎨 Launching R2 Training Studio...")
  print("   A beautiful Apple-style interface for robot training\n")
  launch_ui(host="0.0.0.0", port=args.port, open_browser=True)
