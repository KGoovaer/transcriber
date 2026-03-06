# Transcriber

CLI tool to transcribe audio using open-weight multimodal models (Qwen-Audio, Gemma, etc.) via Ollama or Hugging Face Transformers.

## Install

```bash
python3.11 -m pip install -e ".[dev]"
```

## Usage

```bash
# Record from mic (press Enter to stop)
transcribe

# Transcribe a file
transcribe --file audio.wav

# Choose backend and model
transcribe --backend ollama --model qwen2-audio
transcribe --backend huggingface --model openai/whisper-large-v3

# Save output
transcribe --file audio.wav --output transcript.txt
```

## Configuration

Defaults are stored in `~/.transcriber/config.toml` (auto-created on first run):

```toml
[transcriber]
default_backend = "ollama"
default_model = "qwen2-audio"
```

## Backends

### Ollama

Install Ollama from https://ollama.com, then:

```bash
ollama serve
ollama pull qwen2-audio
```

### Hugging Face

Models are downloaded automatically on first use. Requires `torch` and `transformers`.

## Run Tests

```bash
python3.11 -m pytest -v
```
