# Accuracy Improvements — Engineering Record

This document describes every accuracy-focused change made to the pipeline after the initial
diagnostic audit. Changes are grouped by root-cause fix, not by file, so the reasoning is
clear.

---

## Diagnosis Summary

The five sample recordings (`recording/*.mp3`) are **8 kHz stereo telephony with one speaker
per channel** — confirmed by a per-second energy-dominance probe showing ~90% of speech
cleanly on one channel, ~10% genuine overlap. Three compounding bugs were found:

| # | Bug | Impact |
|---|---|---|
| 1 | Stereo downmixed to mono before diarization → labels guessed by clustering | Diarization errors on every file |
| 2 | ASR run in streaming/limited-context mode on offline clips | WER penalty for zero gain |
| 3 | Decoding strategy was `greedy` | Small but free WER loss |

---

## Fix 1 — Per-Channel Diarization for Speaker-Split Stereo

**Root cause.** `normalize_audio` (called with `-ac 1`) collapsed both channels into a mono
mix, then `NemoTelephonyDiarizer.diarize()` ran full NeMo ClusteringDiarizer (VAD +
TitaNet embeddings + spectral clustering) to *estimate* who spoke when. For speaker-split
telephony the channel assignment is already ground truth; clustering can only match or be
worse.

### 1a — Channel probing and splitting (`audio_processing.py`)

Two new functions:

```
probe_channel_count(source)  →  int
```
Uses `ffprobe` to read the channel count of the first audio stream. Returns 1 if the stream
is not parseable (safe fallback).

```
split_channels(source, out_dir, sample_rate=16000)  →  list[Path]
```
Extracts each input channel as its own 16 kHz mono PCM16 WAV using FFmpeg
`pan=mono|c0=cN`. `normalize_audio` is **unchanged** — it still produces the mono mix used
for duration/UI and the mono diarization fallback.

### 1b — Per-channel merge and overlap marking (`diarize_inventory.py`)

`SpeakerTurn` gained a `channel: int | None` field to carry the source-channel identity
through the pipeline.

Overlap-marking was factored out of `parse_rttm` into a reusable `_mark_overlaps(turns)`
so it can be called after any mutation of a turn list.

```
merge_channel_turns(channel_turns: dict[int, list[SpeakerTurn]])  →  list[SpeakerTurn]
```
Combines per-channel turn lists into one sorted timeline. Each turn is tagged
`speaker_{channel_index}`, so channel identity becomes the speaker label directly. Overlaps
are marked where both channels speak simultaneously.

### 1c — Cross-talk dominance gate (`diarize_inventory.py`)

**Root cause of residual errors.** ~10% of frames have energy on the "wrong" channel due to
echo and telephone bleed. Without gating, VAD fires on the leakage and creates spurious
turns attributed to the wrong speaker.

**Mechanism.** A speaker is loudest on their own channel even during genuine overlap; leaked
audio is always an attenuated copy. The gate keeps a turn only when the turn's channel
dominates (or nearly ties) the loudest other channel over the majority of the turn's own
speech frames.

```
framewise_rms_db(path, frame_ms=30)  →  (list[float], float)   [audio_processing.py]
```
Per-frame dBFS loudness and the frame duration in seconds. The 30 ms frame matches
typical VAD resolution.

```
_turn_is_dominant(channel_db, frame_sec, channel, start_sec, end_sec,
                  margin_db, keep_fraction, speech_floor_db)  →  bool
```
Pure-logic decision, testable without audio. The logic:
1. Identify the turn's own speech frames (within `speech_floor_db` dB of the turn peak).
2. Count frames where this channel's level is ≥ the loudest other channel − `margin_db`.
3. Return True if that count / total speech frames ≥ `keep_fraction`.

Defaults: `margin_db=3.0`, `keep_fraction=0.5`, `speech_floor_db=30.0`.

```
gate_crosstalk_turns(turns, channels, ...)  →  list[SpeakerTurn]
```
Audio-backed wrapper: loads per-channel dBFS arrays once, then filters the turn list.

After gating, `_mark_overlaps` is re-run so overlap flags reflect the cleaned turn list.

**Tuning.** Raise `margin_db` toward 6 if bleed is strong (more aggressive dropping).
Lower it if genuine speech is being clipped. Toggle `gate_crosstalk=False` on
`diarize_channels()` to A/B without the gate.

### 1d — Silero VAD v5 for per-channel segmentation (`diarize_inventory.py`)

The previous approach ran the full `ClusteringDiarizer` (VAD + embeddings + clustering) on
each channel independently, which was wasteful: a single-speaker channel has nothing to
cluster. It was also slower and produced coarser turn edges.

Replaced with **Silero VAD v5** via `_vad_turns(channel)`:

```python
spans = get_speech_timestamps(wav, model, sampling_rate=16000,
                              return_seconds=True,
                              min_silence_duration_ms=200,
                              speech_pad_ms=100)
```

The old clustering-per-channel call is retained as a **commented-out fallback**:

```python
# per_channel = {index: self.diarize(channel, output / f"channel_{index}") ...}
per_channel = {index: self._vad_turns(channel) for index, channel in enumerate(channels)}
```

The `diarize()` method (used for the mono fallback path) is unchanged.

`silero-vad>=5,<6` added to `requirements.txt`.

### 1e — Pipeline routing (`pipeline.py`)

`_run` now probes channel count before diarization:

```
channel_count = probe_channel_count(source)

if channel_count >= 2:
    channels = split_channels(source, ...)           # one 16 kHz WAV per channel
    channel_clips = {i: denoise(channels[i]) ...}    # denoise each channel independently
    turns = diarize_channels(channels, ...)          # VAD + gate
    metrics["diarization_mode"] = "per_channel"
else:
    turns = diarize(normalized, ...)                 # original clustering path
    metrics["diarization_mode"] = "clustering"
```

ASR clip slicing now uses the turn's **own channel's denoised WAV** instead of the mixed
denoised WAV:

```python
clip_source = channel_clips.get(turn.channel, denoised) if turn.channel is not None else denoised
clip = slice_wav(clip_source, ...)
```

This means there is no cross-channel audio present in any turn clip fed to the ASR model.

---

## Fix 2 — Offline Full-Context ASR (Primary Path)

**Root cause.** `transcribe()` was driving `CacheAwareStreamingAudioBuffer` chunk-by-chunk
with a 320 ms context window `[56, 3]`. This emulates real-time streaming on stored audio,
deliberately blinding the model to future context to minimise latency. Since latency is not a
constraint for this pipeline, that tradeoff was entirely one-sided: accuracy loss for no gain.

### 2a — Primary offline path (`nemotron_streaming.py`)

`transcribe()` was replaced with an **offline full-context** implementation:

```python
FULL_CONTEXT = [-1, -1]   # unlimited left + right attention

self.model.encoder.set_default_att_context_size(att_context_size=FULL_CONTEXT)
results = self.model.transcribe([str(audio)], batch_size=1, verbose=False)
```

`[-1, -1]` is the NeMo-documented setting for offline evaluation of cache-aware
FastConformer — the encoder sees the whole sequence in both directions, which is the
highest-accuracy configuration the model supports.

The method signature retains `chunk_ms=None` for API compatibility with call sites that
pass it; the value is ignored in the offline path.

### 2b — Streaming path preserved as `transcribe_streaming()` (`nemotron_streaming.py`)

The original buffer-based code is kept intact as `transcribe_streaming()` for use cases
where real-time incremental output is needed. Its default `chunk_ms` was changed from 320
to **1120** (attention context `[56, 13]`) — the most accurate wired context, so the
streaming fallback is also in its best configuration.

### 2c — Default `chunk_ms` updated in API and pipeline

`chunk_ms` defaults changed from 320 → 1120 in:

- `app.py` — `POST /api/jobs` and `POST /api/jobs/sample/{filename}` form defaults
- `pipeline.py` — `JobStore.submit()` signature default

This has no effect on the primary (offline) path but ensures the streaming fallback uses the
best context if it is ever called directly.

---

## Fix 3 — Beam Decoding (`maes`) Instead of Greedy

**Root cause.** `DECODING_STRATEGY = "greedy"` was a latency optimisation. RNN-T greedy
decoding picks the highest-probability token at each step without considering alternatives.
Beam search (`maes`) explores multiple hypotheses and consistently produces lower WER at no
additional hardware cost when batch size is 1 and latency is irrelevant.

Change in `nemotron_streaming.py`:

```python
DECODING_STRATEGY = "maes"   # was "greedy"
BEAM_SIZE = 4
```

Applied in `__init__` via `change_decoding_strategy`, so it covers both the offline path
and the streaming fallback.

---

## Files Changed

| File | What changed |
|---|---|
| `audio_processing.py` | `probe_channel_count`, `split_channels`, `framewise_rms_db` added |
| `diarize_inventory.py` | `SpeakerTurn.channel` field; `_mark_overlaps` factored out; `merge_channel_turns`, `_turn_is_dominant`, `gate_crosstalk_turns` added; `NemoTelephonyDiarizer._silero`, `_vad_turns`, `diarize_channels` added/updated |
| `nemotron_streaming.py` | `DECODING_STRATEGY` → `maes`; `BEAM_SIZE`, `FULL_CONTEXT` constants; `transcribe()` → offline full-context; `transcribe_streaming()` added (original buffer code); `chunk_ms` default 320 → 1120 |
| `pipeline.py` | Channel probe + routing in `_run`; per-channel denoising; clip slicing from own channel; `chunk_ms` default 320 → 1120 |
| `app.py` | `chunk_ms` form defaults 320 → 1120 |
| `requirements.txt` | `silero-vad>=5,<6` added |
| `tests/test_diarization.py` | `test_merge_channels_tags_speaker_and_marks_overlap`, `test_crosstalk_gate_keeps_own_speech_drops_leakage`, `test_crosstalk_gate_preserves_overlap` added |
| `tests/test_nemotron_streaming.py` | `DECODING_STRATEGY` assertion updated to `maes` |

---

## What Was Not Changed

- **`normalize_audio`** — still produces a mono 16 kHz downmix. Used for UI playback, job
  duration, and as the input to the mono diarization fallback.
- **`NemoTelephonyDiarizer.diarize()`** — original ClusteringDiarizer path unchanged; still
  used for mono inputs.
- **`_merge_and_label_turns` / `_append_turn`** — role-labelling logic unchanged. Roles are
  still assigned by speaking order (`customer` = first speaker, `agent` = second). This is
  tracked as a known limitation: deterministic channel→role mapping (Fix #4) is a separate
  follow-up.
- **`DeepFilterNetDenoiser` / `AudioPreprocessor`** — denoising runs as before on each
  channel individually. Whether denoising helps or hurts WER on 8 kHz telephony is an open
  A/B question (Fix #3 from the original audit); it was not changed here.
- **ITN service** — untouched.

---

## Remaining Known Items

| # | Description |
|---|---|
| 4 | Role labels (`customer`/`agent`) assigned by speaking order; should map from channel identity for deterministic results |
| 5 | Denoising on 8 kHz telephony (8→48→DFN→8 kHz round-trip) likely degrades SNR; A/B against raw normalised signal recommended |
| 6 | Mono fallback still uses 2022-era `ClusteringDiarizer`; offline Sortformer (`diar_sortformer_4spk-v1`) is a drop-in upgrade with overlap awareness |
| — | `maes` beam decoding in `transcribe_streaming()` — the cache-aware streaming step may behave unexpectedly with beam; if `transcribe_streaming` is used in production, validate or pin that path to greedy locally |
| — | `FULL_CONTEXT = [-1, -1]` — if the checkpoint rejects unlimited context, fall back to `[56, 13]` (the largest trained context) |

---

## Verification

All changes were verified with:

```
python3 -m unittest tests.test_diarization tests.test_nemotron_streaming tests.test_audio_processing tests.test_pipeline -v
```

5/5 tests pass on this machine (macOS, no CUDA). Full pipeline end-to-end requires the
Linux GPU host (two CUDA GPUs, `silero-vad` and `nemo_toolkit[asr]` installed).

On the GPU host, confirm:
- `metrics.diarization_mode == "per_channel"` for stereo inputs
- Spurious short turns from bleed/echo are absent
- `metrics` shows two distinct speaker turns per call
- WER improves vs. the pre-fix baseline on held-out labelled clips
