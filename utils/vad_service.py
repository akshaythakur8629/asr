import threading
from pathlib import Path
from typing import Any

class SileroVADService:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def get_model(self):
        with self._lock:
            if self._model is None:
                from silero_vad import load_silero_vad
                self._model = load_silero_vad()
            return self._model

    def get_speech_turns(
        self,
        channel_path: Path,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 100,
        sampling_rate: int = 16000
    ) -> list[dict[str, Any]]:
        """Run Silero VAD on an audio file and return speech timestamps as dict list."""
        from silero_vad import get_speech_timestamps, read_audio
        wav = read_audio(str(channel_path), sampling_rate=sampling_rate)
        spans = get_speech_timestamps(
            wav,
            self.get_model(),
            sampling_rate=sampling_rate,
            return_seconds=True,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        return spans
