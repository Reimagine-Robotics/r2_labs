# R2 Training Studio

Web interface for training robot skills with live progress monitoring.

## Quick Start

```bash
uv run sake training_ui
```

Then open http://localhost:8000 in your browser.

## Features

- Live training progress via WebSocket
- Real-time loss visualization
- Searchable entry filter selection
- Model export to warehouse
- Dark mode support

## Usage

1. **Connect** - Enter training server host:port
2. **Configure** - Set model name, steps, and select entry filters
3. **Train** - Start training and monitor live progress
4. **Export** - Save trained model to warehouse

## UI Preview

<img width="800" alt="Training UI" src="https://github.com/user-attachments/assets/..." />

The interface shows:
- Training configuration panel
- Live phase indicator
- Progress bars (dataset export, training)
- Real-time metrics (steps, loss, speed, ETA)
- Loss chart with smoothing

## Dependencies

Already included in uv environment:
- `fastapi` - Web framework
- `uvicorn` - ASGI server

No additional installation needed.
