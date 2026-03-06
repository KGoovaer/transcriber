import pytest
from transcriber.backends.base import TranscriptionBackend


def test_cannot_instantiate_base_class():
    with pytest.raises(TypeError):
        TranscriptionBackend()


def test_subclass_must_implement_transcribe():
    class IncompleteBackend(TranscriptionBackend):
        pass

    with pytest.raises(TypeError):
        IncompleteBackend()


def test_subclass_with_transcribe_can_be_instantiated():
    class ConcreteBackend(TranscriptionBackend):
        def transcribe(self, audio_path: str) -> str:
            return "hello"

    backend = ConcreteBackend()
    assert backend.transcribe("any.wav") == "hello"
