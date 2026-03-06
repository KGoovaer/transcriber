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
