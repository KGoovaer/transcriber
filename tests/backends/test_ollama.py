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
