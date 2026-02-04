"""Apple-style Training UI for R2 SDK.

A beautiful web-based interface for training robot skills with live progress
monitoring and an intuitive Apple-inspired design.

Usage:
    from r2_labs.sdk.training_ui import launch_ui

    launch_ui()  # Opens in browser automatically
"""

from r2_labs.sdk.training_ui.launch import launch as launch_ui

__all__ = ["launch_ui"]
