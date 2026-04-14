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

## Downloading Wav2Lip weights

The `--model wav2lip` path (Milestone 3b) requires the pretrained checkpoint
`wav2lip_gan.pth` (~420 MB) to exist at `models/wav2lip_gan.pth`. It is not
checked into git. Download it once:

```bash
mkdir -p models
curl -fL -o models/wav2lip_gan.pth \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth
```

If the HuggingFace mirror disappears, search HuggingFace for
`wav2lip_gan.pth` — several community mirrors host the same file.

## Usage (Track B)

```bash
# Passthrough (Milestone 1)
uv run python -m cli.main --video in.mp4 --output out.mp4

# Pipeline with placeholder IdentityModel (Milestone 3a)
uv run python -m cli.main --video in.mp4 --output out.mp4 --lipsync

# Full Wav2Lip lip-sync (Milestone 3b, requires wav2lip_gan.pth)
uv run python -m cli.main \
  --video in.mp4 \
  --audio dubbed.wav \
  --output out.mp4 \
  --lipsync --model wav2lip
```

## Transcription demo (auto-dub sub-project: STT)

The `core.transcription` module exposes a Python API for speech-to-text
via `mlx-whisper`. A standalone demo script transcribes a video end-to-end:

```bash
uv run python examples/transcribe_demo.py <video_path> [<output_stem>]
```

Produces `<stem>.wav` (extracted audio), `<stem>.json` (canonical data
with word-level timestamps + probabilities) and `<stem>.srt` (human-readable
subtitles).

First run downloads the default `medium` Whisper checkpoint (~1.5 GB) to
`~/.cache/huggingface/hub/`. Subsequent runs use the cached weights.

## Translation demo (auto-dub sub-project: translate)

The `core.translation` module translates a Transcription into another
language via the Claude API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/translate_demo.py examples/veo_stt.json en
```

Reads the source JSON (produced by `transcribe_demo.py`), calls Claude
Haiku 4.5 with a structured tool schema that forces a 1:1 segment mapping,
and writes the translated result as `examples/veo_stt.en.json`. Segment
timestamps are preserved; word-level timestamps are dropped in the
translated output.

Default model is `claude-haiku-4-5-20251001`. Cost is on the order of
a fraction of a cent per 10 minutes of transcribed content at current
pricing.
