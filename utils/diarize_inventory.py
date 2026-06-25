"""Standalone NeMo telephony diarization for one normalized recording."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class SpeakerTurn:
    speaker: str; start_sec: float; end_sec: float; overlap_flag: bool = False; channel: int | None = None

def _mark_overlaps(turns: list[SpeakerTurn]) -> None:
    """Flag any turn that intersects in time with a different speaker's turn."""
    for i, turn in enumerate(turns):
        turn.overlap_flag = any(i != j and turn.speaker != other.speaker and min(turn.end_sec, other.end_sec) > max(turn.start_sec, other.start_sec) for j, other in enumerate(turns))

def parse_rttm(path: Path) -> list[SpeakerTurn]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER": turns.append(SpeakerTurn(p[7], float(p[3]), float(p[3]) + float(p[4])))
    turns.sort(key=lambda x: (x.start_sec, x.end_sec, x.speaker))
    _mark_overlaps(turns)
    return turns

def _turn_is_dominant(channel_db: list[list[float]], frame_sec: float, channel: int, start_sec: float, end_sec: float, margin_db: float, keep_fraction: float, speech_floor_db: float) -> bool:
    """True if this turn is the channel's own speech, not another channel's leakage.

    Over the turn's own speech frames, the turn is kept when its channel stays within
    `margin_db` of (or above) the loudest other channel for at least `keep_fraction` of
    them. A speaker is loudest on their own channel even during overlap, so genuine
    speech survives while echo/bleed (where another channel dominates) is dropped.
    """
    length = min(len(c) for c in channel_db)
    a = max(0, int(start_sec / frame_sec)); b = min(length, int(round(end_sec / frame_sec)))
    if b <= a: return True
    mine = channel_db[channel][a:b]; peak = max(mine)
    speech = [i for i, v in enumerate(mine) if v > peak - speech_floor_db]
    if not speech: return True
    owned = sum(1 for i in speech if mine[i] >= max((channel_db[c][a + i] for c in range(len(channel_db)) if c != channel), default=-120.0) - margin_db)
    return owned / len(speech) >= keep_fraction

def gate_crosstalk_turns(turns: list[SpeakerTurn], channels: list[Path], margin_db: float = 3.0, keep_fraction: float = 0.5, speech_floor_db: float = 30.0) -> list[SpeakerTurn]:
    """Drop per-channel turns dominated by another channel's leakage (echo/cross-talk)."""
    if len(channels) < 2: return turns
    from .audio_processing import framewise_rms_db
    channel_db: list[list[float]] = []; frame_sec = 0.03
    for path in channels:
        db, frame_sec = framewise_rms_db(path); channel_db.append(db)
    return [t for t in turns if t.channel is None or not (0 <= t.channel < len(channel_db)) or _turn_is_dominant(channel_db, frame_sec, t.channel, t.start_sec, t.end_sec, margin_db, keep_fraction, speech_floor_db)]

def merge_channel_turns(channel_turns: dict[int, list[SpeakerTurn]]) -> list[SpeakerTurn]:
    """Combine per-channel speech turns into one timeline, tagging each by its channel.

    Each channel of speaker-split telephony is one speaker, so the channel index *is*
    the speaker label; overlaps fall out where two channels speak at once.
    """
    merged: list[SpeakerTurn] = []
    for index in sorted(channel_turns):
        for turn in channel_turns[index]:
            merged.append(SpeakerTurn(f"speaker_{index}", turn.start_sec, turn.end_sec, channel=index))
    merged.sort(key=lambda x: (x.start_sec, x.end_sec, x.speaker))
    _mark_overlaps(merged)
    return merged

def _config(manifest: Path, output: Path, device: str, max_speakers: int):
    from omegaconf import OmegaConf
    return OmegaConf.create({"name":"ClusterDiarizer","num_workers":1,"sample_rate":16000,"batch_size":64,"device":device,"verbose":False,"diarizer":{"manifest_filepath":str(manifest),"out_dir":str(output),"oracle_vad":False,"collar":0.25,"ignore_overlap":True,"vad":{"model_path":"vad_multilingual_marblenet","external_vad_manifest":None,"parameters":{"window_length_in_sec":0.15,"shift_length_in_sec":0.01,"smoothing":"median","overlap":0.5,"onset":0.1,"offset":0.1,"pad_onset":0.1,"pad_offset":0.0,"min_duration_on":0.0,"min_duration_off":0.2,"filter_speech_first":True}},"speaker_embeddings":{"model_path":"titanet_large","parameters":{"window_length_in_sec":[1.5,1.25,1.0,0.75,0.5],"shift_length_in_sec":[0.75,0.625,0.5,0.375,0.25],"multiscale_weights":[1,1,1,1,1],"save_embeddings":False}},"clustering":{"parameters":{"oracle_num_speakers":False,"max_num_speakers":max_speakers,"enhanced_count_thres":80,"max_rp_threshold":0.25,"sparse_search_volume":30,"maj_vote_spk_count":False,"chunk_cluster_count":50,"embeddings_per_chunk":10000}}}})

class NemoTelephonyDiarizer:
    def __init__(self, device="cuda:1", max_speakers=2): self.device=device; self.max_speakers=max_speakers; self._vad_model=None

    def _silero(self):
        """Lazily load Silero VAD v5 once. Light enough to run on CPU; gives crisp telephony boundaries."""
        if self._vad_model is None:
            from silero_vad import load_silero_vad
            self._vad_model = load_silero_vad()
        return self._vad_model

    def _vad_turns(self, channel: Path) -> list[SpeakerTurn]:
        """Segment one channel into speech turns with Silero VAD (speaker is assigned later by channel)."""
        from silero_vad import get_speech_timestamps, read_audio
        wav = read_audio(str(channel), sampling_rate=16000)
        spans = get_speech_timestamps(wav, self._silero(), sampling_rate=16000, return_seconds=True, min_silence_duration_ms=200, speech_pad_ms=100)
        return [SpeakerTurn("speaker", float(s["start"]), float(s["end"])) for s in spans]

    def diarize(self, audio: Path, output: Path) -> list[SpeakerTurn]:
        import torch
        torch.cuda.set_device(self.device)
        from nemo.collections.asr.models import ClusteringDiarizer
        output.mkdir(parents=True, exist_ok=True); manifest = output / "manifest.json"
        row={"audio_filepath":str(audio.resolve()),"offset":0,"duration":None,"label":"infer","text":"-","num_speakers":None,"rttm_filepath":None,"uem_filepath":None}
        manifest.write_text(json.dumps(row)+"\n", encoding="utf-8")
        model=ClusteringDiarizer(cfg=_config(manifest, output, self.device, self.max_speakers)).to(self.device); model.diarize()
        rttm=output/"pred_rttms"/f"{audio.stem}.rttm"; return parse_rttm(rttm) if rttm.exists() else []

    def diarize_channels(self, channels: list[Path], output: Path, gate_crosstalk: bool = True) -> list[SpeakerTurn]:
        """Diarize speaker-split telephony by channel: VAD-segment each channel, then merge.

        Per-channel turns are collapsed to one speaker id per channel, so any spurious
        intra-channel clustering split is overridden by the channel's ground-truth identity.
        Cross-talk gating then drops turns that are really another channel's leakage.
        """
        output.mkdir(parents=True, exist_ok=True)
        # VAD-only per channel: crisper turn edges and far faster than running full clustering per channel.
        # Previous clustering-per-channel segmentation (kept for reference / fallback):
        # per_channel = {index: self.diarize(channel, output / f"channel_{index}") for index, channel in enumerate(channels)}
        per_channel = {index: self._vad_turns(channel) for index, channel in enumerate(channels)}
        turns = merge_channel_turns(per_channel)
        if gate_crosstalk:
            turns = gate_crosstalk_turns(turns, channels)
            _mark_overlaps(turns)  # re-evaluate overlaps after leakage turns are removed
        return turns

def turns_to_dicts(turns): return [asdict(turn) for turn in turns]
