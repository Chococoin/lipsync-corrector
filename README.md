# lipsync-corrector

Side project. Two tracks:

- **Track A (Milestone 0):** face-swap quick win. A single script that swaps a reference face into a video.
- **Track B (Milestone 1+):** lip-sync corrector for auto-dubbed YouTube videos.

See `docs/superpowers/specs/` for the design spec.

## Requirements

- macOS on Apple Silicon (tested on M4).
- `ffmpeg` (install with `brew install ffmpeg`).
- `uv` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Setup

```bash
uv sync
# Download the inswapper model (see docs/milestones/milestone-0.md)
```

## Usage (Track A)

```bash
uv run python swap.py \
  --face examples/reference.jpg \
  --video examples/input.mp4 \
  --output examples/output.mp4 \
  --max-seconds 15
```
