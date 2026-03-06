# Transcriber

CLI tool to transcribe audio using open-weight multimodal models (Qwen-Audio, Gemma, etc.) via Ollama or Hugging Face Transformers.

## Install

```bash
python3.11 -m pip install -e ".[dev]"
```

### faster-whisper (optional, recommended for long recordings)

```bash
python3.11 -m pip install -e ".[faster-whisper]"
```

If the `transcribe` command is not found after installing, add the Python user bin directory to your PATH:

```bash
echo 'export PATH="$HOME/Library/Python/3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
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

# Use faster-whisper for long recordings (~4x faster than huggingface)
transcribe --backend faster-whisper --file recording.wav
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

Models are downloaded from Hugging Face on first use and cached locally (`~/.cache/huggingface/`). All inference runs fully locally — no internet connection needed after the initial download. Requires `torch` and `transformers`.

### faster-whisper

Same Whisper models as Hugging Face but runs via CTranslate2, making it ~4x faster. Useful for long recordings. Models are also cached locally after the first download.

```bash
transcribe --backend faster-whisper --file recording.wav
```

## Run Tests

```bash
python3.11 -m pytest -v
```
