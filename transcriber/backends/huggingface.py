import os
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transcriber.backends.base import TranscriptionBackend


class HuggingFaceBackend(TranscriptionBackend):
    def __init__(self, model: str = "openai/whisper-large-v3"):
        self.model_name = model
        self._processor = None
        self._model = None

    def _load(self):
        if self._processor is None:
            print(f"Loading model '{self.model_name}' (this may take a moment on first run)...")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, dtype=torch.float32)

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load()
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        generated = self._model.generate(**inputs, task="transcribe")
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]
