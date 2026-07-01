"""Single-worker temporary processing pipeline."""
from __future__ import annotations
import os
import shutil, threading, time, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from .audio_processing import normalize_audio, probe_channel_count, slice_wav, split_channels, wav_duration
from .denoise_service import AudioPreprocessor
from .vad_service import SileroVADService
from .lid_service import SpeechBrainLIDService
from .transcript_render import render_markdown_transcript


DEPLOY_NEMOTRON = os.environ.get("DEPLOY_NEMOTRON", "true").lower() == "true"
DEPLOY_INDIC = os.environ.get("DEPLOY_INDIC", "false").lower() == "true"

def resolve_model(requested_model: str) -> str:
    """Resolve the model name based on deployment environment variables."""
    # If both models are deployed (e.g. port 8001)
    if DEPLOY_NEMOTRON and DEPLOY_INDIC:
        if requested_model in {"credresolve:v1", "sarga:v1.1"}:
            return "credresolve:v1"
        return "sarga:v1"
    # If only Nemotron is deployed (e.g. port 8000)
    elif DEPLOY_NEMOTRON:
        return "sarga:v1"
    # If only Indic Conformer is deployed (e.g. port 8002)
    elif DEPLOY_INDIC:
        return "credresolve:v1"
    return "sarga:v1"

NemoTelephonyDiarizer = None
NemotronStreamingASR = None
SpeakerTurn = None

try:
    from .diarize_inventory import NemoTelephonyDiarizer, SpeakerTurn
except ImportError as e:
    print(f"Warning: Failed to import NeMo Diarizer dependencies: {e}")

if DEPLOY_NEMOTRON:
    try:
        from nemotron_model.model import NemotronStreamingASR
    except ImportError as e:
        print(f"Warning: Failed to import NeMo ASR dependencies: {e}")

if SpeakerTurn is None:
    class SpeakerTurn:
        def __init__(self, speaker, start_sec, end_sec, overlap_flag=False, channel=None):
            self.speaker = speaker
            self.start_sec = start_sec
            self.end_sec = end_sec
            self.overlap_flag = overlap_flag
            self.channel = channel

IndicStreamingASR = None
if DEPLOY_INDIC:
    try:
        from indic_model.model import IndicStreamingASR
    except ImportError as e:
        print(f"Warning: Failed to import Indic model dependencies: {e}")


class JobStore:
    def __init__(self, root=Path("/tmp/nemotron-test"), ttl_seconds=259200):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="nemotron-job")
        self.preprocessor = None
        self.diarizer = None
        self.asr = None
        self.asr_indic = None
        self.vad_service = None
        self.lid_service = None
        self.diarizer_lock = threading.Lock()
        self.asr_lock = threading.Lock()
        self.asr_indic_lock = threading.Lock()

    def submit(self, source: Path, filename: str, language="hi-IN", chunk_ms=1120, itn_backend="custom", model="sarga:v1", diarize=True, denoise=True) -> dict:
        model = resolve_model(model)
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True)
        suffix = Path(filename).suffix or ".webm"
        saved = job_dir / f"input{suffix}"
        shutil.copyfile(source, saved)
        job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0, "filename": filename, "language": language, "chunk_ms": chunk_ms, "itn_backend": itn_backend, "model": model, "created_at": time.time(), "result": None, "error": None, "diarize": diarize, "denoise": denoise}
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run, job_id, saved, language, chunk_ms, itn_backend, model, diarize, denoise)
        return self.public(job_id)

    def update(self, job_id, **changes):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(changes)

    def public(self, job_id):
        self.cleanup()
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def _silero(self):
        """Helper to get the cached Silero VAD model."""
        if self.vad_service is None:
            self.vad_service = SileroVADService()
        return self.vad_service.get_model()

    def _ensure_models(self):
        with self.lock:
            if self.preprocessor is None:
                self.preprocessor = AudioPreprocessor()
            if self.vad_service is None:
                self.vad_service = SileroVADService()
            if self.lid_service is None:
                self.lid_service = SpeechBrainLIDService()
                self.lid_service.load_model()
            if self.diarizer is None and NemoTelephonyDiarizer is not None:
                import torch
                diarizer_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
                self.diarizer = NemoTelephonyDiarizer(device=diarizer_device, max_speakers=2)

            if DEPLOY_NEMOTRON:
                if self.asr is None and NemotronStreamingASR is not None:
                    self.asr = NemotronStreamingASR(device="cuda:0")
            if DEPLOY_INDIC:
                if self.asr_indic is None and IndicStreamingASR is not None:
                    self.asr_indic = IndicStreamingASR(device="cuda:0")
    def _run(self,job_id,source,language,chunk_ms,itn_backend,model="sarga:v1",diarize=True,denoise=True):
        job_dir=self.root/job_id; metrics={}; started=time.perf_counter(); biasing_applied=False
        try:
            self.update(job_id,status="running",stage="normalizing",progress=5); t=time.perf_counter()
            normalized=normalize_audio(source,job_dir/"normalized.wav"); metrics["normalize_seconds"]=round(time.perf_counter()-t,3)
            self.update(job_id,stage="loading_models",progress=15); self._ensure_models()
            
            if denoise:
                self.update(job_id,stage="denoising",progress=25); t=time.perf_counter(); denoised=self.preprocessor.denoise_wav(normalized,job_dir/"denoised.wav"); metrics["denoise_seconds"]=round(time.perf_counter()-t,3)
            else:
                denoised=normalized
                metrics["denoise_seconds"]=0.0
            channel_count=probe_channel_count(source); channel_clips={}
            
            turns = None
            if self.diarizer is not None:
                # Use NeMo diarizer for chunking in BOTH modes:
                # - diarize=True  → speaker turns used with Customer/Agent labeling
                # - diarize=False → speaker turns used only for chunking; output is flat text
                self.update(job_id, stage="diarizing", progress=45); t = time.perf_counter()
                with self.diarizer_lock:
                    if channel_count >= 2:
                        channels = split_channels(source, job_dir/"channels")
                        channel_clips = {index: self.preprocessor.denoise_wav(path, job_dir/f"channel_{index}_denoised.wav") for index, path in enumerate(channels)}
                        turns = self.diarizer.diarize_channels(channels, job_dir/"diarization")
                        metrics["diarization_mode"] = "per_channel"; metrics["channels"] = channel_count
                    else:
                        turns = self.diarizer.diarize(normalized, job_dir/"diarization")
                        metrics["diarization_mode"] = "clustering"
                metrics["diarization_seconds"] = round(time.perf_counter()-t, 3)

                if not diarize and turns:
                    # Flat mode: re-label all turns as the same speaker so they merge
                    # into one flat block; NeMo boundaries are kept for chunking only.
                    for turn in turns:
                        turn.speaker = "speaker_0"
            else:
                metrics["diarization_seconds"] = 0.0
                metrics["diarization_mode"] = "none"

            if not turns:
                # Fallback when diarizer is unavailable: 25-second time-based slices.
                FLAT_CHUNK_SEC = 25.0
                total_dur = wav_duration(normalized)
                if total_dur <= FLAT_CHUNK_SEC:
                    turns = [SpeakerTurn("speaker_0", 0.0, total_dur, False)]
                else:
                    turns = []
                    t_start = 0.0
                    while t_start < total_dur:
                        t_end = min(t_start + FLAT_CHUNK_SEC, total_dur)
                        turns.append(SpeakerTurn("speaker_0", t_start, t_end, False))
                        t_start = t_end


            if language == "majority" and self.lid_service is not None:
                self.update(job_id, stage="detecting_language", progress=50)
                fallback_langs = {"as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}
                supported = getattr(self.asr_indic, "supported_languages", fallback_langs) if self.asr_indic is not None else fallback_langs
                language = self.lid_service.vote_file_language(
                    turns=turns,
                    channel_clips=channel_clips,
                    denoised_path=denoised,
                    job_dir=job_dir,
                    supported_languages=supported,
                    default_lang="hi-IN"
                )

            results=[]; asr_started=time.perf_counter()
            detected_languages=[]
            asr_lock_to_use = self.asr_indic_lock if model in {"sarga:v1.1", "credresolve:v1"} else self.asr_lock
            with asr_lock_to_use:
                metrics["biasing"]={"applied":False,"reason":"disabled"}
                    
                try:
                    for index,turn in enumerate(turns):
                        self.update(job_id,stage=f"transcribing turn {index+1}/{len(turns)}",progress=55+int(40*(index/max(1,len(turns)))))
                        clip_source=channel_clips.get(turn.channel,denoised) if turn.channel is not None else denoised
                        clip=slice_wav(clip_source,job_dir/f"turn_{index:04d}.wav",turn.start_sec,turn.end_sec)
                        
                        if model in {"sarga:v1.1", "credresolve:v1"}:
                            if self.asr_indic is None:
                                raise RuntimeError(f"Model {model} (Indic Conformer) is not deployed on this server.")
                            text, resolved_lang = self.asr_indic.transcribe_with_lang(clip,language=language,chunk_ms=chunk_ms)
                        else:
                            if self.asr is None:
                                raise RuntimeError(f"Model {model} (Nemotron) is not deployed on this server.")
                            text, resolved_lang = self.asr.transcribe_with_lang(clip,language=language,chunk_ms=chunk_ms)
                            
                        if text.strip():
                            normalized_turn=_normalize_turn(text,resolved_lang,itn_backend)
                            results.append({"speaker":turn.speaker,"start_sec":round(turn.start_sec,3),"end_sec":round(turn.end_sec,3),"overlap_flag":turn.overlap_flag,"text":text,"language_id":resolved_lang,**normalized_turn})
                            detected_languages.append(resolved_lang)
                finally:
                    if biasing_applied:
                        try:self.asr.reset_decoding()
                        except Exception:pass  # never let cleanup mask the job result
            results=_merge_and_label_turns(results)
            
            if detected_languages:
                from collections import Counter
                overall_lang = Counter(detected_languages).most_common(1)[0][0]
            else:
                overall_lang = language if language != "auto" else "hi-IN"
                
            metrics["asr_seconds"]=round(time.perf_counter()-asr_started,3); metrics["total_seconds"]=round(time.perf_counter()-started,3); metrics["audio_seconds"]=round(wav_duration(normalized),3)
            if diarize:
                transcript_str = "\n".join(f"{x['speaker']}: {x['canonical_text']}" for x in results)
                raw_transcript_str = "\n".join(f"{x['speaker']}: {x['text']}" for x in results)
            else:
                # Flat mode: no speaker labels, just plain concatenated text
                transcript_str = " ".join(x['canonical_text'] for x in results)
                raw_transcript_str = " ".join(x['text'] for x in results)
            result={
                "turns":results,
                "transcript":transcript_str,
                "raw_transcript":raw_transcript_str,
                "markdown":render_markdown_transcript(results),
                "itn_backend":itn_backend,
                "language_id":overall_lang,
                "metrics":metrics,
                "diarize":diarize,
                "original_url":f"/api/jobs/{job_id}/audio/normalized",
                "denoised_url":f"/api/jobs/{job_id}/audio/denoised"
            }
            self.update(job_id,status="complete",stage="complete",progress=100,result=result)
        except Exception as exc:
            self.update(job_id,status="failed",stage="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
    def audio_path(self,job_id,kind):
        if kind not in {"normalized","denoised"}:return None
        path=self.root/job_id/f"{kind}.wav"; return path if path.exists() else None
    def cleanup(self):
        cutoff=time.time()-self.ttl
        with self.lock:
            expired=[
                job_id for job_id,job in self.jobs.items()
                if job["created_at"]<cutoff and job["status"] in {"complete", "failed"}
            ]
        for job_id in expired:
            shutil.rmtree(self.root/job_id,ignore_errors=True)
            with self.lock:self.jobs.pop(job_id,None)


def _normalize_turn(text: str, language: str, backend: str) -> dict[str, Any]:
    """Run final-only offline ITN without ever blocking transcript delivery."""
    if backend == "none":
        return {"canonical_text": text, "display_text": text, "spans": [], "itn_deferred": True}
    lang = language.split("-", 1)[0] if language and language != "auto" else "hi"
    try:
        from itn_service.runtime.offline_normalizer import OfflineComparison, normalize_offline_text
        result = normalize_offline_text(text, lang_hint=lang, backend=backend)
        if isinstance(result, OfflineComparison):
            custom = result.custom_result
            return {
                "canonical_text": custom.canonical_text, "display_text": custom.display_text,
                "spans": [span.model_dump(mode="json") for span in custom.spans],
                "itn_deferred": custom.deferred, "custom_canonical_text": custom.canonical_text,
                "nemo_canonical_text": result.nemo_result.canonical_text,
                "outputs_equal": result.outputs_equal,
            }
        return {"canonical_text": result.canonical_text, "display_text": result.display_text,
                "spans": [span.model_dump(mode="json") for span in result.spans],
                "itn_deferred": result.deferred}
    except Exception as exc:
        return {"canonical_text": text, "display_text": text, "spans": [],
                "itn_deferred": True, "itn_error": f"{type(exc).__name__}: {exc}"}


def _merge_and_label_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop empty turns, merge consecutive diarization fragments, and label roles."""
    merged: list[dict[str, Any]] = []
    for original in turns:
        if not str(original.get("text", "")).strip():
            continue
        turn = dict(original)
        if merged and merged[-1]["speaker_id"] == turn["speaker"]:
            _append_turn(merged[-1], turn)
            continue
        turn["speaker_id"] = turn["speaker"]
        merged.append(turn)
    role_by_speaker: dict[str, str] = {}
    for turn in merged:
        speaker_id = turn["speaker_id"]
        if speaker_id not in role_by_speaker:
            role_by_speaker[speaker_id] = "customer" if not role_by_speaker else "agent"
        turn["speaker"] = role_by_speaker[speaker_id]
    return merged


def _append_turn(target: dict[str, Any], source: dict[str, Any]) -> None:
    raw_shift = len(target["text"]) + 1
    target["text"] = f"{target['text']} {source['text']}".strip()
    for field in ("canonical_text", "display_text", "custom_canonical_text", "nemo_canonical_text"):
        if field in target or field in source:
            target[field] = f"{target.get(field, '')} {source.get(field, '')}".strip()
    shifted = []
    for span in source.get("spans", []):
        span = dict(span)
        if span.get("start") is not None: span["start"] += raw_shift
        if span.get("end") is not None: span["end"] += raw_shift
        shifted.append(span)
    target.setdefault("spans", []).extend(shifted)
    target["end_sec"] = source["end_sec"]
    target["overlap_flag"] = target.get("overlap_flag", False) or source.get("overlap_flag", False)
    target["itn_deferred"] = target.get("itn_deferred", False) or source.get("itn_deferred", False)
    if source.get("itn_error"):
        target["itn_error"] = source["itn_error"]
