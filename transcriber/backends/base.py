from abc import ABC, abstractmethod


class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file at audio_path and return transcript string."""
