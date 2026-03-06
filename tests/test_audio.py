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
