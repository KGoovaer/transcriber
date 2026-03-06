import base64
import httpx
from transcriber.backends.base import TranscriptionBackend


class OllamaBackend(TranscriptionBackend):
    def __init__(self, model: str = "qwen2-audio", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Transcribe this audio. Return only the transcript text.",
                    "images": [audio_b64],
                }
            ],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        raise RuntimeError(
                            f"Model '{self.model}' not found. Try: ollama pull {self.model}"
                        ) from e
                    raise
                return response.json()["message"]["content"]
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Ollama not reachable at {self.base_url}. Is it running? Try: ollama serve"
            ) from e
