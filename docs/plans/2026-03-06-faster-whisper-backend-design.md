# Design: faster-whisper Backend

## Summary

Add `faster-whisper` as a third backend option alongside `ollama` and `huggingface`. It uses CTranslate2 under the hood, delivering ~4x faster inference than the standard transformers implementation with identical accuracy. Useful for long recordings (e.g. 1 hour).

## Architecture

New class `FasterWhisperBackend` in `transcriber/backends/faster_whisper.py`, following the same pattern as `HuggingFaceBackend`:

- `_load()`: instantiate `WhisperModel(model_name, device="auto", compute_type="int8")`
- `transcribe()`: call `model.transcribe(audio_path)`, join segment texts

## Changes

- `transcriber/backends/faster_whisper.py` — new backend class
- `transcriber/cli.py` — register `"faster-whisper"` in `BACKENDS`, wire up in `main()`
- `pyproject.toml` — add `faster-whisper` optional dependency under `[project.optional-dependencies]` as a new `faster-whisper` extra
- Default backend in config remains `huggingface`

## Error Handling

Raise `RuntimeError` with a helpful message if `faster-whisper` is not installed (import error).

## Testing

Add `tests/backends/test_faster_whisper.py` mirroring `test_huggingface.py`.
