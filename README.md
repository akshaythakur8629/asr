# Nemotron Streaming ASR Lab

Local FastAPI test harness for `nvidia/nemotron-3.5-asr-streaming-0.6b`, DeepFilterNet3, and offline NeMo telephony diarization.

## Pipeline

1. Browser microphone recording, upload, or included `recording/` sample.
2. FFmpeg normalization to 16 kHz mono PCM WAV.
3. DeepFilterNet enhancement of the complete signal without VAD, preserving timestamps.
4. Diarization on GPU 1. Stereo telephony is diarized **per channel** (one speaker per channel, segmented with Silero VAD v5 for crisp turn edges, overlaps marked where both channels speak). A cross-talk gate then drops turns that are really another channel's echo/leakage, keeping genuine speech and overlap. Mono inputs fall back to NeMo clustering with `vad_multilingual_marblenet` + `titanet_large`.
5. Offline full-context Nemotron transcription of each speaker turn on GPU 0 (unlimited right-context + `maes` beam search for highest accuracy; the cache-aware streaming path is kept as a fallback). Turns are sliced from that turn's own denoised channel (no cross-talk) for stereo, or the denoised mono mix for mono.
6. Final-only offline ITN per speaker turn, using the custom production backend by default.

Jobs run one at a time and artifacts are stored under `/tmp/nemotron-test` for one hour.

## Setup

The host needs two CUDA-visible GPUs, FFmpeg, libsndfile1, and a recent CUDA-compatible PyTorch build. DeepFilterNet may require Rust/Cargo while installing its native dependency.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Select the offline ITN backend in the frontend; each turn shows normalized text, raw ASR when changed, and span provenance. The first job downloads and loads the ASR, VAD, and speaker models, so it takes longer than later jobs.

## Notes

- Hindi `hi-IN` is the default. Transcription is offline full-context by default; the `chunk_ms` control only affects the streaming fallback (default 1120 ms = the most accurate wired context).
- Stereo telephony uses channel-based diarization (ground-truth speaker per channel); mono inputs estimate one or two speakers via clustering. Overlap is marked but not source-separated.
- The included files have `.mp3` names but contain 8 kHz stereo PCM; FFmpeg handles them correctly.
- Run lightweight tests with `python3 -m unittest discover -s tests -v`.
# asr
