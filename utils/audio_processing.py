"""Standalone audio normalization and DeepFilterNet helpers."""
from __future__ import annotations
import subprocess, threading, wave, sys, types

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

from math import gcd
from pathlib import Path
import numpy as np

TARGET_SAMPLE_RATE = 16000

def normalize_audio(source: Path, destination: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Try high-quality soxr resampling first
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-af", "aresample=resampler=soxr", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(destination)], check=True)
    except subprocess.CalledProcessError:
        # Fallback to standard resampling if libsoxr is not compiled in FFmpeg
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(destination)], check=True)
    return destination

def probe_channel_count(source: Path) -> int:
    """Return the channel count of the first audio stream via ffprobe (1 if unknown)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else 1

def split_channels(source: Path, out_dir: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> list[Path]:
    """Extract each input channel as its own mono PCM16 WAV at the target rate.

    Speaker-split telephony keeps one party per channel, so per-channel files give
    ground-truth speaker separation that downmix-then-cluster diarization can only guess at.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    for index in range(probe_channel_count(source)):
        destination = out_dir / f"channel_{index}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-af", f"pan=mono|c0=c{index}", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(destination)],
            check=True,
        )
        destinations.append(destination)
    return destinations

def read_pcm16_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        return f.readframes(f.getnframes()), f.getframerate()

def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sample_rate); f.writeframes(pcm)
    return path

def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f: return f.getnframes() / float(f.getframerate())

def framewise_rms_db(path: Path, frame_ms: int = 30) -> tuple[list[float], float]:
    """Return per-frame loudness in dBFS and the frame duration in seconds.

    Used to gate cross-channel leakage: a frame belongs to whichever channel is loudest.
    """
    pcm, sr = read_pcm16_wav(path)
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frame = max(1, int(sr * frame_ms / 1000))
    frame_sec = frame / float(sr)
    n = audio.size // frame
    if n == 0: return [-120.0], frame_sec
    blocks = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    return (20.0 * np.log10(np.maximum(rms, 1e-6))).tolist(), frame_sec

def slice_wav(source: Path, destination: Path, start_sec: float, end_sec: float) -> Path:
    pcm, sr = read_pcm16_wav(source); audio = np.frombuffer(pcm, dtype="<i2")
    start, end = max(0, round(start_sec * sr)), min(audio.size, round(end_sec * sr))
    return write_pcm16_wav(destination, audio[start:end].tobytes(), sr)

from .denoise_service import DeepFilterNetDenoiser, AudioPreprocessor

