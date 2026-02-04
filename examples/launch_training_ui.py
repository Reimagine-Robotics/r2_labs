#!/usr/bin/env python3
"""Launch the R2 Training Studio UI.

This is a beautiful, Apple-style web interface for configuring and monitoring
robot skill training. No notebooks required - just pure app-like experience.

Usage:
    python launch_training_ui.py

Then open http://localhost:8000 in your browser (opens automatically).
"""

from r2_labs.sdk.training_ui import launch_ui

if __name__ == "__main__":
    print("\n🎨 Launching R2 Training Studio...")
    print("   A beautiful Apple-style interface for robot training\n")
    launch_ui(host="0.0.0.0", port=8000, open_browser=True)
