from __future__ import annotations
import os
import sys
import click
from transcriber.audio import AudioCapture
from transcriber.backends.ollama import OllamaBackend
from transcriber.backends.huggingface import HuggingFaceBackend
from transcriber.backends.faster_whisper import FasterWhisperBackend
from transcriber.config import load_config
from transcriber.output import write_output


BACKENDS = {"ollama", "huggingface", "faster-whisper"}


@click.command()
@click.option("--file", "-f", "input_file", default=None, help="Audio file to transcribe (skips mic recording).")
@click.option("--backend", "-b", default=None, help="Backend to use: ollama, huggingface, or faster-whisper.")
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

    if backend == "ollama":
        transcription_backend = OllamaBackend(model=model)
    elif backend == "faster-whisper":
        transcription_backend = FasterWhisperBackend(model=model)
    else:
        transcription_backend = HuggingFaceBackend(model=model)

    transcription_backend._load()

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

    try:
        transcript = transcription_backend.transcribe(audio_path)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    write_output(transcript, output_file=output_file)
