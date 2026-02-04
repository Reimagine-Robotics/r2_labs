#!/usr/bin/env python3
"""Launch the R2 Training Studio UI."""

import webbrowser
from pathlib import Path


def launch(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True):
  """Launch the training UI web application.

  Args:
      host: Host to bind the server to (default: 0.0.0.0)
      port: Port to run the server on (default: 8000)
      open_browser: If True, automatically open the browser (default: True)
  """
  import uvicorn

  print("=" * 60)
  print("🚀 R2 Training Studio")
  print("=" * 60)
  print(f"\n📡 Server: http://localhost:{port}")
  print("\n💡 Open the URL above in your browser to access the UI")
  print("\n⌨️  Press Ctrl+C to stop the server\n")

  if open_browser:
    # Open browser after a short delay
    import threading

    def open_in_browser():
      import time

      time.sleep(1.5)
      webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_in_browser, daemon=True).start()

  # Get the app module path
  app_path = str(Path(__file__).parent / "app.py")

  uvicorn.run(
      "r2_labs.sdk.training_ui.app:app",
      host=host,
      port=port,
      reload=False,
      log_level="info",
  )


if __name__ == "__main__":
  launch()
