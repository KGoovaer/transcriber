# faster-whisper Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `faster-whisper` as a third backend option (`--backend faster-whisper`) for ~4x faster transcription of long recordings.

**Architecture:** New `FasterWhisperBackend` class in `transcriber/backends/faster_whisper.py`, following the same `_load()` / `transcribe()` pattern as `HuggingFaceBackend`. The CLI wires it up alongside the existing backends.

**Tech Stack:** `faster-whisper` (pip package), CTranslate2 under the hood, same Whisper model weights.

---

### Task 1: Add faster-whisper optional dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add the optional extra**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
faster-whisper = [
    "faster-whisper>=1.0",
]
```

**Step 2: Install it**

```bash
python3.11 -m pip install -e ".[faster-whisper]"
```

Expected: installs `faster-whisper` and `ctranslate2`.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add faster-whisper optional dependency"
```

---

### Task 2: Implement FasterWhisperBackend

**Files:**
- Create: `transcriber/backends/faster_whisper.py`
- Test: `tests/backends/test_faster_whisper.py`

**Step 1: Write the failing tests**

Create `tests/backends/test_faster_whisper.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
from transcriber.backends.faster_whisper import FasterWhisperBackend

FIXTURE_WAV = "tests/fixtures/silence.wav"


def test_transcribe_returns_string():
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " hello world"
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    with patch("transcriber.backends.faster_whisper.WhisperModel", return_value=mock_model):
        backend = FasterWhisperBackend(model="tiny")
        backend._load()
        result = backend.transcribe(FIXTURE_WAV)

    assert result == "hello world"


def test_transcribe_file_not_found():
    mock_model = MagicMock()

    with patch("transcriber.backends.faster_whisper.WhisperModel", return_value=mock_model):
        backend = FasterWhisperBackend(model="tiny")
        backend._load()
        with pytest.raises(FileNotFoundError):
            backend.transcribe("nonexistent.wav")


def test_load_raises_on_missing_package():
    with patch.dict("sys.modules", {"faster_whisper": None}):
        with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
            import importlib
            import transcriber.backends.faster_whisper as m
            importlib.reload(m)
            m.FasterWhisperBackend(model="tiny")._load()
```

**Step 2: Run tests to verify they fail**

```bash
python3.11 -m pytest tests/backends/test_faster_whisper.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — file doesn't exist yet.

**Step 3: Implement the backend**

Create `transcriber/backends/faster_whisper.py`:

```python
import os
from transcriber.backends.base import TranscriptionBackend

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class FasterWhisperBackend(TranscriptionBackend):
    def __init__(self, model: str = "large-v3"):
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            if WhisperModel is None:
                raise RuntimeError(
                    "faster-whisper is not installed. Run: pip install 'transcriber[faster-whisper]'"
                )
            print(f"Loading model '{self.model_name}' (this may take a moment on first run)...")
            self._model = WhisperModel(self.model_name, device="auto", compute_type="int8")

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load()
        segments, _ = self._model.transcribe(audio_path)
        return "".join(segment.text for segment in segments).strip()
```

**Step 4: Run tests to verify they pass**

```bash
python3.11 -m pytest tests/backends/test_faster_whisper.py -v
```

Expected: all 3 tests PASS.

**Step 5: Commit**

```bash
git add transcriber/backends/faster_whisper.py tests/backends/test_faster_whisper.py
git commit -m "feat: add FasterWhisperBackend"
```

---

### Task 3: Wire faster-whisper into the CLI

**Files:**
- Modify: `transcriber/cli.py`

**Step 1: Update the CLI**

In `transcriber/cli.py`, make these changes:

1. Add import at the top (after `HuggingFaceBackend`):
```python
from transcriber.backends.faster_whisper import FasterWhisperBackend
```

2. Update `BACKENDS`:
```python
BACKENDS = {"ollama", "huggingface", "faster-whisper"}
```

3. Update the `--backend` help string:
```python
@click.option("--backend", "-b", default=None, help="Backend to use: ollama, huggingface, or faster-whisper.")
```

4. Extend the backend selection block:
```python
if backend == "ollama":
    transcription_backend = OllamaBackend(model=model)
elif backend == "faster-whisper":
    transcription_backend = FasterWhisperBackend(model=model)
else:
    transcription_backend = HuggingFaceBackend(model=model)
```

**Step 2: Run existing CLI tests**

```bash
python3.11 -m pytest tests/test_cli.py -v
```

Expected: all tests PASS.

**Step 3: Smoke test manually**

```bash
transcribe --backend faster-whisper --file <any .wav file>
```

Expected: transcript printed to stdout.

**Step 4: Commit**

```bash
git add transcriber/cli.py
git commit -m "feat: wire faster-whisper backend into CLI"
```

---

### Task 4: Update README

**Files:**
- Modify: `README.md`

**Step 1: Add faster-whisper install instructions**

Under `## Install`, add a new subsection:

```markdown
### faster-whisper (optional, recommended for long recordings)

```bash
python3.11 -m pip install -e ".[faster-whisper]"
```
```

**Step 2: Add usage example**

Under `## Usage`, add:

```markdown
# Use faster-whisper for long recordings
transcribe --backend faster-whisper --file recording.wav
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document faster-whisper backend"
```
