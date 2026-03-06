# Transcriber Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python CLI that records mic audio (or reads a file) and transcribes it using pluggable open-weight multimodal models via Ollama or Hugging Face Transformers.

**Architecture:** Audio is captured to a temp WAV file, then passed to a `TranscriptionBackend` (abstract base class). Two concrete backends exist: `OllamaBackend` (HTTP API) and `HuggingFaceBackend` (local transformers). The CLI wires everything together via `click`.

**Tech Stack:** Python 3.11+, click, sounddevice, soundfile, httpx, transformers, torch, pytest, pytest-mock

---

### Task 1: Project scaffold — pyproject.toml and package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `transcriber/__init__.py`
- Create: `transcriber/cli.py`
- Create: `transcriber/audio.py`
- Create: `transcriber/output.py`
- Create: `transcriber/backends/__init__.py`
- Create: `transcriber/backends/base.py`
- Create: `transcriber/backends/ollama.py`
- Create: `transcriber/backends/huggingface.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/silence.wav` (generated in step below)

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "transcriber"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "sounddevice>=0.4",
    "soundfile>=0.12",
    "httpx>=0.27",
    "transformers>=4.40",
    "torch>=2.2",
    "tomli>=2.0; python_version < '3.11'",
]

[project.scripts]
transcribe = "transcriber.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create the package skeleton**

Create each file listed above as an empty file (just `pass` or a blank `__init__.py`):

```bash
mkdir -p transcriber/backends tests/fixtures
touch transcriber/__init__.py
touch transcriber/backends/__init__.py
touch tests/__init__.py
```

**Step 3: Generate a 1-second silent WAV fixture for tests**

```python
# Run this once in a Python shell to create tests/fixtures/silence.wav
import soundfile as sf
import numpy as np
sf.write("tests/fixtures/silence.wav", np.zeros(16000, dtype="float32"), 16000)
```

Or via bash (after installing soundfile):
```bash
python -c "import soundfile as sf, numpy as np; sf.write('tests/fixtures/silence.wav', np.zeros(16000, dtype='float32'), 16000)"
```

**Step 4: Install the package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: installs without errors, `transcribe` command available.

**Step 5: Commit**

```bash
git add pyproject.toml transcriber/ tests/
git commit -m "feat: scaffold project structure and pyproject.toml"
```

---

### Task 2: TranscriptionBackend abstract base class

**Files:**
- Create: `transcriber/backends/base.py`
- Create: `tests/backends/test_base.py`

**Step 1: Write the failing test**

Create `tests/backends/__init__.py` (empty), then `tests/backends/test_base.py`:

```python
import pytest
from transcriber.backends.base import TranscriptionBackend


def test_cannot_instantiate_base_class():
    with pytest.raises(TypeError):
        TranscriptionBackend()


def test_subclass_must_implement_transcribe():
    class IncompleteBackend(TranscriptionBackend):
        pass

    with pytest.raises(TypeError):
        IncompleteBackend()


def test_subclass_with_transcribe_can_be_instantiated():
    class ConcreteBackend(TranscriptionBackend):
        def transcribe(self, audio_path: str) -> str:
            return "hello"

    backend = ConcreteBackend()
    assert backend.transcribe("any.wav") == "hello"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/backends/test_base.py -v
```

Expected: FAIL — `ImportError: cannot import name 'TranscriptionBackend'`

**Step 3: Implement**

```python
# transcriber/backends/base.py
from abc import ABC, abstractmethod


class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file at audio_path and return transcript string."""
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/backends/test_base.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add transcriber/backends/base.py tests/backends/
git commit -m "feat: add TranscriptionBackend abstract base class"
```

---

### Task 3: OllamaBackend

**Files:**
- Create: `transcriber/backends/ollama.py`
- Create: `tests/backends/test_ollama.py`

**Step 1: Write the failing tests**

```python
# tests/backends/test_ollama.py
import base64
import pytest
from unittest.mock import MagicMock, patch
from transcriber.backends.ollama import OllamaBackend

FIXTURE_WAV = "tests/fixtures/silence.wav"


def test_transcribe_returns_string(tmp_path):
    wav = tmp_path / "test.wav"
    wav.write_bytes(open(FIXTURE_WAV, "rb").read())

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "hello world"}
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        backend = OllamaBackend(model="qwen2-audio", base_url="http://localhost:11434")
        result = backend.transcribe(str(wav))

    assert result == "hello world"


def test_transcribe_raises_on_connection_error(tmp_path):
    import httpx
    wav = tmp_path / "test.wav"
    wav.write_bytes(open(FIXTURE_WAV, "rb").read())

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.ConnectError("refused")

        backend = OllamaBackend(model="qwen2-audio", base_url="http://localhost:11434")
        with pytest.raises(RuntimeError, match="Ollama not reachable"):
            backend.transcribe(str(wav))


def test_transcribe_raises_on_model_not_found(tmp_path):
    import httpx
    wav = tmp_path / "test.wav"
    wav.write_bytes(open(FIXTURE_WAV, "rb").read())

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock(status_code=404, text="model not found")
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        backend = OllamaBackend(model="qwen2-audio", base_url="http://localhost:11434")
        with pytest.raises(RuntimeError, match="not found"):
            backend.transcribe(str(wav))
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/backends/test_ollama.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Implement**

```python
# transcriber/backends/ollama.py
import base64
import httpx
from transcriber.backends.base import TranscriptionBackend


class OllamaBackend(TranscriptionBackend):
    def __init__(self, model: str = "qwen2-audio", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Transcribe this audio. Return only the transcript text.",
                    "images": [audio_b64],
                }
            ],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        raise RuntimeError(
                            f"Model '{self.model}' not found. Try: ollama pull {self.model}"
                        ) from e
                    raise
                return response.json()["message"]["content"]
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}. Is it running? Try: ollama serve"
            ) from e
```

**Step 4: Run tests**

```bash
pytest tests/backends/test_ollama.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add transcriber/backends/ollama.py tests/backends/test_ollama.py
git commit -m "feat: add OllamaBackend with error handling"
```

---

### Task 4: HuggingFaceBackend

**Files:**
- Create: `transcriber/backends/huggingface.py`
- Create: `tests/backends/test_huggingface.py`

**Step 1: Write the failing tests**

```python
# tests/backends/test_huggingface.py
import pytest
from unittest.mock import MagicMock, patch
from transcriber.backends.huggingface import HuggingFaceBackend

FIXTURE_WAV = "tests/fixtures/silence.wav"


def test_transcribe_returns_string():
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_inputs = MagicMock()
    mock_processor.return_value = mock_inputs
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["hello world"]

    with patch("transcriber.backends.huggingface.AutoProcessor.from_pretrained", return_value=mock_processor), \
         patch("transcriber.backends.huggingface.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=mock_model):

        backend = HuggingFaceBackend(model="openai/whisper-tiny")
        result = backend.transcribe(FIXTURE_WAV)

    assert result == "hello world"


def test_transcribe_file_not_found():
    with patch("transcriber.backends.huggingface.AutoProcessor.from_pretrained", return_value=MagicMock()), \
         patch("transcriber.backends.huggingface.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=MagicMock()):

        backend = HuggingFaceBackend(model="openai/whisper-tiny")
        with pytest.raises(FileNotFoundError):
            backend.transcribe("nonexistent.wav")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/backends/test_huggingface.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Implement**

```python
# transcriber/backends/huggingface.py
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transcriber.backends.base import TranscriptionBackend


class HuggingFaceBackend(TranscriptionBackend):
    def __init__(self, model: str = "openai/whisper-large-v3"):
        self.model_name = model
        self._processor = None
        self._model = None

    def _load(self):
        if self._processor is None:
            print(f"Loading model '{self.model_name}' (this may take a moment on first run)...")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)

    def transcribe(self, audio_path: str) -> str:
        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load()
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        generated = self._model.generate(**inputs)
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]
```

**Step 4: Run tests**

```bash
pytest tests/backends/test_huggingface.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add transcriber/backends/huggingface.py tests/backends/test_huggingface.py
git commit -m "feat: add HuggingFaceBackend"
```

---

### Task 5: AudioCapture

**Files:**
- Create: `transcriber/audio.py`
- Create: `tests/test_audio.py`

**Step 1: Write the failing tests**

```python
# tests/test_audio.py
import os
import shutil
import pytest
from transcriber.audio import AudioCapture

FIXTURE_WAV = "tests/fixtures/silence.wav"


def test_from_file_returns_path_to_existing_file(tmp_path):
    dest = tmp_path / "input.wav"
    shutil.copy(FIXTURE_WAV, dest)
    capture = AudioCapture()
    result = capture.from_file(str(dest))
    assert result == str(dest)


def test_from_file_raises_if_file_not_found():
    capture = AudioCapture()
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        capture.from_file("does_not_exist.wav")


def test_from_file_raises_if_no_audio():
    """Empty file should raise ValueError."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"")
        name = f.name
    try:
        capture = AudioCapture()
        with pytest.raises(Exception):
            capture.from_file(name)
    finally:
        os.unlink(name)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_audio.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Implement**

```python
# transcriber/audio.py
import os
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000


class AudioCapture:
    def from_file(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        # Validate it's readable audio
        try:
            data, _ = sf.read(path, dtype="float32")
        except Exception as e:
            raise ValueError(f"Cannot read audio file '{path}': {e}") from e
        if len(data) == 0:
            raise ValueError(f"Audio file '{path}' contains no audio data.")
        return path

    def from_microphone(self) -> str:
        """Record from mic until user presses Enter. Returns path to temp WAV file."""
        print("Recording... press Enter to stop.")
        frames = []
        stop_event = threading.Event()

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)

        with stream:
            input()  # block until Enter
            stop_event.set()

        if not frames:
            raise ValueError("No audio captured. Is your microphone working?")

        audio = np.concatenate(frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)
        return tmp.name
```

**Step 4: Run tests**

```bash
pytest tests/test_audio.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add transcriber/audio.py tests/test_audio.py
git commit -m "feat: add AudioCapture for mic and file input"
```

---

### Task 6: Output handler

**Files:**
- Create: `transcriber/output.py`
- Create: `tests/test_output.py`

**Step 1: Write the failing tests**

```python
# tests/test_output.py
import os
import pytest
from transcriber.output import write_output


def test_prints_to_stdout(capsys):
    write_output("hello world", output_file=None)
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"


def test_writes_to_file(tmp_path):
    out = tmp_path / "transcript.txt"
    write_output("hello world", output_file=str(out))
    assert out.read_text().strip() == "hello world"


def test_writes_to_file_and_prints(tmp_path, capsys):
    out = tmp_path / "transcript.txt"
    write_output("hello world", output_file=str(out))
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
    assert out.read_text().strip() == "hello world"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_output.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Implement**

```python
# transcriber/output.py
from __future__ import annotations


def write_output(transcript: str, output_file: str | None) -> None:
    print(transcript)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")
```

**Step 4: Run tests**

```bash
pytest tests/test_output.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add transcriber/output.py tests/test_output.py
git commit -m "feat: add output handler"
```

---

### Task 7: Configuration loader

**Files:**
- Create: `transcriber/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing tests**

```python
# tests/test_config.py
import os
import pytest
from unittest.mock import patch
from transcriber.config import load_config, DEFAULT_CONFIG


def test_returns_defaults_when_no_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config = load_config(str(config_path))
    assert config["default_backend"] == "ollama"
    assert config["default_model"] == "qwen2-audio"


def test_overrides_defaults_with_file_values(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[transcriber]\ndefault_model = "gemma3"\n')
    config = load_config(str(config_path))
    assert config["default_model"] == "gemma3"
    assert config["default_backend"] == "ollama"  # still default


def test_creates_config_file_if_missing(tmp_path):
    config_path = tmp_path / "config.toml"
    load_config(str(config_path))
    assert config_path.exists()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Implement**

```python
# transcriber/config.py
from __future__ import annotations
import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_CONFIG = {
    "default_backend": "ollama",
    "default_model": "qwen2-audio",
}

DEFAULT_CONFIG_TOML = """\
[transcriber]
default_backend = "ollama"
default_model = "qwen2-audio"
"""


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.expanduser("~/.transcriber/config.toml")

    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG_TOML)
        return dict(DEFAULT_CONFIG)

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    section = data.get("transcriber", {})
    return {**DEFAULT_CONFIG, **section}
```

**Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add transcriber/config.py tests/test_config.py
git commit -m "feat: add config loader with defaults"
```

---

### Task 8: CLI entry point

**Files:**
- Create: `transcriber/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing tests**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from transcriber.cli import main

FIXTURE_WAV = "tests/fixtures/silence.wav"


def test_cli_with_file_and_ollama(tmp_path):
    runner = CliRunner()
    out_file = tmp_path / "out.txt"

    mock_backend = MagicMock()
    mock_backend.transcribe.return_value = "hello from ollama"

    with patch("transcriber.cli.OllamaBackend", return_value=mock_backend), \
         patch("transcriber.cli.AudioCapture") as mock_capture_cls:
        mock_capture = MagicMock()
        mock_capture.from_file.return_value = FIXTURE_WAV
        mock_capture_cls.return_value = mock_capture

        result = runner.invoke(main, [
            "--file", FIXTURE_WAV,
            "--backend", "ollama",
            "--model", "qwen2-audio",
            "--output", str(out_file),
        ])

    assert result.exit_code == 0
    assert "hello from ollama" in result.output
    assert out_file.read_text().strip() == "hello from ollama"


def test_cli_fails_gracefully_on_backend_error():
    runner = CliRunner()
    mock_backend = MagicMock()
    mock_backend.transcribe.side_effect = RuntimeError("Ollama not reachable")

    with patch("transcriber.cli.OllamaBackend", return_value=mock_backend), \
         patch("transcriber.cli.AudioCapture") as mock_capture_cls:
        mock_capture = MagicMock()
        mock_capture.from_file.return_value = FIXTURE_WAV
        mock_capture_cls.return_value = mock_capture

        result = runner.invoke(main, ["--file", FIXTURE_WAV, "--backend", "ollama"])

    assert result.exit_code != 0
    assert "Ollama not reachable" in result.output


def test_cli_unknown_backend_fails():
    runner = CliRunner()
    result = runner.invoke(main, ["--file", FIXTURE_WAV, "--backend", "unknown"])
    assert result.exit_code != 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL — `ImportError`

**Step 3: Implement**

```python
# transcriber/cli.py
from __future__ import annotations
import os
import sys
import click
from transcriber.audio import AudioCapture
from transcriber.backends.ollama import OllamaBackend
from transcriber.backends.huggingface import HuggingFaceBackend
from transcriber.config import load_config
from transcriber.output import write_output


BACKENDS = {"ollama", "huggingface"}


@click.command()
@click.option("--file", "-f", "input_file", default=None, help="Audio file to transcribe (skips mic recording).")
@click.option("--backend", "-b", default=None, help="Backend to use: ollama or huggingface.")
@click.option("--model", "-m", default=None, help="Model name to use for transcription.")
@click.option("--output", "-o", "output_file", default=None, help="Save transcript to this file.")
def main(input_file, backend, model, output_file):
    """Transcribe audio using open-weight models."""
    config = load_config()
    backend = backend or config["default_backend"]
    model = model or config["default_model"]

    if backend not in BACKENDS:
        click.echo(f"Error: Unknown backend '{backend}'. Choose from: {', '.join(BACKENDS)}", err=True)
        sys.exit(1)

    capture = AudioCapture()
    try:
        if input_file:
            audio_path = capture.from_file(input_file)
            temp_file = None
        else:
            audio_path = capture.from_microphone()
            temp_file = audio_path
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if backend == "ollama":
        transcription_backend = OllamaBackend(model=model)
    else:
        transcription_backend = HuggingFaceBackend(model=model)

    try:
        transcript = transcription_backend.transcribe(audio_path)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    write_output(transcript, output_file=output_file)
```

**Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: 3 passed

**Step 5: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass

**Step 6: Commit**

```bash
git add transcriber/cli.py tests/test_cli.py
git commit -m "feat: add CLI entry point wiring all components together"
```

---

### Task 9: README

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

```markdown
# Transcriber

CLI tool to transcribe audio using open-weight multimodal models (Qwen-Audio, Gemma, etc.) via Ollama or Hugging Face Transformers.

## Install

```bash
pip install -e ".[dev]"
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
pytest -v
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install, usage, and backend instructions"
```

---

### Task 10: Final verification

**Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass, no warnings about missing imports.

**Step 2: Verify the CLI is installed**

```bash
transcribe --help
```

Expected: shows help text with all options.

**Step 3: Smoke test with a file (optional, requires Ollama running)**

```bash
transcribe --file tests/fixtures/silence.wav --backend ollama --model qwen2-audio
```

Expected: prints a transcript (or empty string for silence).
