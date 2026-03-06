from abc import ABC, abstractmethod


class TranscriptionBackend(ABC):
    def _load(self) -> None:
        """Pre-load any model or resources. No-op by default."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file at audio_path and return transcript string."""
