# Transcriber — Design Document

Date: 2026-03-06

## Overview

A Python CLI application that records audio from a microphone (or reads an audio file) and transcribes it using open-weight multimodal models such as Qwen-Audio or Gemma. The backend is pluggable: models can be served via Ollama or loaded directly via Hugging Face Transformers.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   CLI (cli.py)                   │
│  transcribe [--model MODEL] [--backend BACKEND]  │
│             [--file FILE] [--output FILE]        │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   AudioCapture      │  sounddevice / soundfile
          │  (mic or file)      │
          └──────────┬──────────┘
                     │ WAV file path
          ┌──────────▼──────────┐
          │  Backend Interface  │  abstract base class
          │  transcribe(audio)  │
          └────────┬────────────┘
           ┌───────┴────────┐
  ┌────────▼───────┐  ┌─────▼──────────────┐
  │  OllamaBackend │  │  HuggingFaceBackend │
  │  (HTTP API)    │  │  (transformers)     │
  └────────────────┘  └────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Output Handler    │  stdout + optional file
          └─────────────────────┘
```

## Components

- **CLI (`cli.py`)** — `click`-based entry point. Parses arguments, orchestrates audio capture, backend selection, and output.
- **AudioCapture (`audio.py`)** — Records from microphone via `sounddevice` until user presses Enter; saves to a temp WAV file. Also accepts a file path directly.
- **TranscriptionBackend (`backends/base.py`)** — Abstract base class with a single method: `transcribe(audio_path: str) -> str`.
- **OllamaBackend (`backends/ollama.py`)** — Calls the Ollama HTTP API with the audio file. Targets multimodal models like `qwen2-audio`.
- **HuggingFaceBackend (`backends/huggingface.py`)** — Loads model and processor via `transformers`, runs local inference.
- **Output Handler (`output.py`)** — Prints transcript to stdout; optionally writes to a file if `--output` is specified.

## CLI Interface

```bash
# Record from mic with defaults
transcribe

# Specify model and backend
transcribe --model qwen2-audio --backend ollama
transcribe --model openai/whisper-large-v3 --backend huggingface

# Transcribe an existing file
transcribe --file audio.mp3

# Save output to file
transcribe --output transcript.txt

# Combined
transcribe --file meeting.wav --model qwen2-audio --backend ollama --output notes.txt
```

## Data Flow

1. CLI starts; if no `--file` given, prints `"Recording... press Enter to stop"`
2. `AudioCapture` records mic via `sounddevice` into a buffer
3. User presses Enter — recording stops, buffer saved as a temp WAV file
4. Selected backend receives the WAV path, sends to model, returns transcript string
5. Transcript printed to stdout; if `--output` given, also written to that file
6. Temp WAV file deleted

## Configuration

Defaults stored in `~/.transcriber/config.toml` (created on first run) and overridable via CLI flags or environment variables:

- `default_backend`: `ollama`
- `default_model`: `qwen2-audio`

## Error Handling

| Scenario | Behaviour |
|---|---|
| Ollama not running | Clear message: "Ollama not reachable. Is it running? Try: ollama serve" |
| Model not pulled | "Model 'qwen2-audio' not found. Try: ollama pull qwen2-audio" |
| No audio captured | Warn and exit cleanly |
| Input file not found | Validate path before starting, fail fast with clear message |
| HuggingFace model download | Show download progress indicator |

## Testing

- Unit tests for each backend using a fixture WAV file
- Unit test for `AudioCapture` file-input path (mic path tested manually)
- Integration test: end-to-end with a short WAV file against a live backend (skipped in CI if backend unavailable)
- Tools: `pytest`, `pytest-mock`

## Project Structure

```
transcriber/
├── transcriber/
│   ├── __init__.py
│   ├── cli.py
│   ├── audio.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ollama.py
│   │   └── huggingface.py
│   └── output.py
├── tests/
├── docs/plans/
├── pyproject.toml
└── README.md
```

## Dependencies

- `click` — CLI framework
- `sounddevice` — microphone recording
- `soundfile` — WAV encoding
- `httpx` — Ollama HTTP calls
- `transformers` + `torch` — HuggingFace backend
- `tomllib` / `tomli` — config file parsing
- `pytest`, `pytest-mock` — testing
