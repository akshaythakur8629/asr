import threading
import types
import sys
from math import gcd
from pathlib import Path
import numpy as np

# Monkeypatch missing torchaudio.backend for compatibility with newer torchaudio versions in DeepFilterNet
try:
    import torchaudio.backend
except ImportError:
    class AudioMetaData:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int, bits_per_sample: int, encoding: str):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    common_module = types.ModuleType("torchaudio.backend.common")
    common_module.AudioMetaData = AudioMetaData
    sys.modules["torchaudio.backend.common"] = common_module
    backend_module = types.ModuleType("torchaudio.backend")
    sys.modules["torchaudio.backend"] = backend_module

from .audio_processing import read_pcm16_wav, write_pcm16_wav


class DeepFilterNetDenoiser:
    def __init__(self):
        import torch
        import df.enhance
        df.enhance.get_device = lambda: torch.device("cpu")
        from df.enhance import init_df
        self.torch = torch
        self.model, self.state, _ = init_df()
        self.lock = threading.Lock()
        value = getattr(self.state, "sr", lambda: 48000)
        self.sample_rate = int(value() if callable(value) else value)
        if self.sample_rate != 48000:
            raise RuntimeError(f"DeepFilterNet sample rate is {self.sample_rate}, expected 48000")

    def process(self, pcm: bytes) -> bytes:
        from df.enhance import enhance
        audio = np.frombuffer(pcm, dtype="<i2")
        if not audio.size:
            return b""
        tensor = self.torch.from_numpy((audio.astype(np.float32) / 32768.0).reshape(1, -1))
        with self.lock, self.torch.inference_mode():
            cleaned = enhance(self.model, self.state, tensor)
        return np.clip(cleaned.detach().cpu().numpy().reshape(-1) * 32768, -32768, 32767).astype("<i2").tobytes()


class AudioPreprocessor:
    """Whole-recording denoising; no VAD, preserving diarization timestamps."""
    def __init__(self):
        self.denoiser = DeepFilterNetDenoiser()

    @staticmethod
    def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        if source_sr == target_sr:
            return audio
        from scipy.signal import resample_poly
        d = gcd(source_sr, target_sr)
        return np.clip(np.rint(resample_poly(audio.astype(np.float32), target_sr // d, source_sr // d)), -32768, 32767).astype("<i2")

    def denoise_wav(self, source: Path, destination: Path) -> Path:
        pcm, sr = read_pcm16_wav(source)
        original = np.frombuffer(pcm, dtype="<i2")
        audio48 = self._resample(original, sr, self.denoiser.sample_rate)
        if audio48.size == 0:
            clean48 = np.empty(0, dtype="<i2")
        else:
            chunk_size = 5 * 60 * self.denoiser.sample_rate
            clean_chunks = []
            for i in range(0, audio48.size, chunk_size):
                chunk = audio48[i:i+chunk_size]
                clean_chunk = np.frombuffer(self.denoiser.process(chunk.tobytes()), dtype="<i2")
                clean_chunks.append(clean_chunk)
            clean48 = np.concatenate(clean_chunks)
        clean = self._resample(clean48, self.denoiser.sample_rate, sr)
        if clean.size < original.size:
            clean = np.pad(clean, (0, original.size - clean.size))
        return write_pcm16_wav(destination, clean[:original.size].tobytes(), sr)
