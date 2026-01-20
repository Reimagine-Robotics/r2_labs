# r2_labs

Client-side SDK for working with R2 Sidekicks.

For more details on how to download and run, see our [Notion page][notion].

[notion]: https://www.notion.so/User-Guide-2e758b7397038047a2b6f6714b94d3c0

## Setup

This package uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and sync

```bash
git clone https://github.com/Reimagine-Robotics/r2_labs.git
cd r2_labs
uv sync
```

### Run scripts

```bash
uv run python -c "import r2_labs; print('ok')"
```

## Jupyter kernel

To use r2_labs in Jupyter notebooks, register a kernel that runs from the uv environment.

### Create the kernel spec

Create a directory for the kernel:

```bash
mkdir -p ~/Library/Jupyter/kernels/r2-labs  # macOS
mkdir -p ~/.local/share/jupyter/kernels/r2-labs  # Linux
```

Create `kernel.json` in that directory with the following content, replacing `/path/to/r2_labs` with the absolute path to your cloned repository:

```json
{
  "argv": [
    "uv",
    "run",
    "--project",
    "/path/to/r2_labs",
    "python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "R2 Labs",
  "language": "python",
  "env": {
    "LOGURU_LEVEL": "WARNING"
  }
}
```

### Use the kernel

The "R2 Labs" kernel will now appear in Jupyter, JupyterLab, and VS Code notebook kernel pickers.
