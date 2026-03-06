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
