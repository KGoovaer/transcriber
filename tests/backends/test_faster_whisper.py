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
