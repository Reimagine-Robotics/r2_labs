# 🎨 R2 Training Studio

A beautiful, Apple-style web interface for training robot skills. No notebooks required - pure app-like experience with modern design.

## ✨ Features

- **🎯 Modern Apple Design** - Clean, minimal UI inspired by Apple's design system
- **🔴 Live Progress** - Real-time training metrics via WebSocket
- **📊 Loss Visualization** - Beautiful charts showing training progress
- **🔍 Smart Filter Selection** - Searchable dropdown for data warehouse entries
- **⚡ Fast & Responsive** - Built with FastAPI and vanilla JS
- **🎨 Color-Coded Phases** - Visual feedback for each training stage

## 🚀 Quick Start

### Launch the UI

```python
from r2_labs.sdk.training_ui import launch_ui

launch_ui()  # Opens browser automatically
```

Or run the standalone script:

```bash
python r2_labs/examples/launch_training_ui.py
```

Then open http://localhost:8000 in your browser.

## 🎮 Usage

1. **Connect** - Enter your training server host and port
2. **Configure** - Set model name, training steps, and select entry filters
3. **Train** - Click start and watch the live progress
4. **Monitor** - View real-time metrics and loss curves
5. **Export** - Save your trained model to the warehouse

## 📦 Dependencies

The UI requires:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `websockets` - For live updates

Install with:
```bash
pip install fastapi uvicorn websockets
```

## 🎨 Design Philosophy

Inspired by Apple's design language:
- San Francisco font family
- Subtle shadows and rounded corners
- Smooth transitions and animations
- Color-coded status indicators
- Minimal, focused interface

## 🛠️ Architecture

```
training_ui/
├── app.py          # FastAPI backend with WebSocket
├── launch.py       # Simple launcher
├── static/
│   ├── index.html  # Main UI
│   ├── styles.css  # Apple-style CSS
│   └── app.js      # Frontend logic
```

**Backend**: Python FastAPI server that connects to R2 training server
**Frontend**: Pure HTML/CSS/JS with WebSocket for live updates
**No frameworks**: Vanilla JS keeps it fast and simple

## 🎯 Key Features

### Searchable Entry Filters
- Type to search data warehouse entries
- Select from dropdown suggestions
- Add custom glob patterns
- Multi-select with visual tags

### Live Progress Monitoring
- WebSocket updates every second
- Phase indicators with colors
- Progress bars for export and training
- Real-time metrics (steps, loss, speed, ETA)

### Loss Chart
- Canvas-based rendering
- Smooth curves
- Auto-scaling
- Apple-style minimalism

Enjoy building amazing robot skills! 🤖✨
