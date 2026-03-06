import os
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000


class AudioCapture:
    def from_file(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            data, _ = sf.read(path, dtype="float32")
        except Exception as e:
            raise ValueError(f"Cannot read audio file '{path}': {e}") from e
        if len(data) == 0:
            raise ValueError(f"Audio file '{path}' contains no audio data.")
        return path

    def from_microphone(self) -> str:
        """Record from mic until user presses Enter. Returns path to temp WAV file."""
        print("Recording... press Enter to stop.")
        frames = []
        stop_event = threading.Event()

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)

        with stream:
            input()
            stop_event.set()

        if not frames:
            raise ValueError("No audio captured. Is your microphone working?")

        audio = np.concatenate(frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, SAMPLE_RATE)
        return tmp.name
