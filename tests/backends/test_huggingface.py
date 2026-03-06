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
