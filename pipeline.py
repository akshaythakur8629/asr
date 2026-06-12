"""Single-worker temporary processing pipeline."""
from __future__ import annotations
import shutil, threading, time, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from audio_processing import AudioPreprocessor, normalize_audio, probe_channel_count, slice_wav, split_channels, wav_duration
from diarize_inventory import NemoTelephonyDiarizer
from nemotron_streaming import NemotronStreamingASR
from transcript_render import render_markdown_transcript

class JobStore:
    def __init__(self, root=Path("/tmp/nemotron-test"), ttl_seconds=3600):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.ttl=ttl_seconds
        self.jobs:dict[str,dict[str,Any]]={}; self.lock=threading.Lock(); self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="nemotron-job")
        self.preprocessor=None; self.diarizer=None; self.asr=None
    def submit(self, source:Path, filename:str, language="hi-IN", chunk_ms=1120, itn_backend="custom", biasing_context=None)->dict:
        job_id=uuid.uuid4().hex; job_dir=self.root/job_id; job_dir.mkdir(parents=True)
        suffix=Path(filename).suffix or ".webm"; saved=job_dir/f"input{suffix}"; shutil.copyfile(source,saved)
        job={"id":job_id,"status":"queued","stage":"queued","progress":0,"filename":filename,"language":language,"chunk_ms":chunk_ms,"itn_backend":itn_backend,"biasing_context":biasing_context,"created_at":time.time(),"result":None,"error":None}
        with self.lock:self.jobs[job_id]=job
        self.executor.submit(self._run,job_id,saved,language,chunk_ms,itn_backend,biasing_context); return self.public(job_id)
    def update(self,job_id,**changes):
        with self.lock:self.jobs[job_id].update(changes)
    def public(self,job_id):
        self.cleanup();
        with self.lock:
            job=self.jobs.get(job_id)
            return dict(job) if job else None
    def _ensure_models(self):
        if self.preprocessor is None:self.preprocessor=AudioPreprocessor()
        if self.diarizer is None:self.diarizer=NemoTelephonyDiarizer(device="cuda:1",max_speakers=2)
        if self.asr is None:self.asr=NemotronStreamingASR(device="cuda:0")
    def _run(self,job_id,source,language,chunk_ms,itn_backend,biasing_context=None):
        job_dir=self.root/job_id; metrics={}; started=time.perf_counter(); biasing_applied=False
        try:
            self.update(job_id,status="running",stage="normalizing",progress=5); t=time.perf_counter()
            normalized=normalize_audio(source,job_dir/"normalized.wav"); metrics["normalize_seconds"]=round(time.perf_counter()-t,3)
            self.update(job_id,stage="loading_models",progress=15); self._ensure_models()
            metrics["biasing"]=self._configure_biasing(language,biasing_context,job_dir)
            biasing_applied=bool(metrics["biasing"] and metrics["biasing"].get("applied"))
            self.update(job_id,stage="denoising",progress=25); t=time.perf_counter(); denoised=self.preprocessor.denoise_wav(normalized,job_dir/"denoised.wav"); metrics["denoise_seconds"]=round(time.perf_counter()-t,3)
            channel_count=probe_channel_count(source); channel_clips={}
            self.update(job_id,stage="diarizing",progress=45); t=time.perf_counter()
            if channel_count>=2:
                channels=split_channels(source,job_dir/"channels")
                channel_clips={index:self.preprocessor.denoise_wav(path,job_dir/f"channel_{index}_denoised.wav") for index,path in enumerate(channels)}
                turns=self.diarizer.diarize_channels(channels,job_dir/"diarization"); metrics["diarization_mode"]="per_channel"; metrics["channels"]=channel_count
            else:
                turns=self.diarizer.diarize(normalized,job_dir/"diarization"); metrics["diarization_mode"]="clustering"
            metrics["diarization_seconds"]=round(time.perf_counter()-t,3)
            if not turns:
                from diarize_inventory import SpeakerTurn
                turns=[SpeakerTurn("speaker_0",0.0,wav_duration(normalized),False)]
            results=[]; asr_started=time.perf_counter()
            for index,turn in enumerate(turns):
                self.update(job_id,stage=f"transcribing turn {index+1}/{len(turns)}",progress=55+int(40*(index/max(1,len(turns)))))
                clip_source=channel_clips.get(turn.channel,denoised) if turn.channel is not None else denoised
                clip=slice_wav(clip_source,job_dir/f"turn_{index:04d}.wav",turn.start_sec,turn.end_sec)
                text=self.asr.transcribe(clip,language=language,chunk_ms=chunk_ms)
                if text.strip():
                    normalized_turn=_normalize_turn(text,language,itn_backend)
                    results.append({"speaker":turn.speaker,"start_sec":round(turn.start_sec,3),"end_sec":round(turn.end_sec,3),"overlap_flag":turn.overlap_flag,"text":text,**normalized_turn})
            results=_merge_and_label_turns(results)
            metrics["asr_seconds"]=round(time.perf_counter()-asr_started,3); metrics["total_seconds"]=round(time.perf_counter()-started,3); metrics["audio_seconds"]=round(wav_duration(normalized),3)
            result={"turns":results,"transcript":"\n".join(f"{x['speaker']}: {x['canonical_text']}" for x in results),"raw_transcript":"\n".join(f"{x['speaker']}: {x['text']}" for x in results),"markdown":render_markdown_transcript(results),"itn_backend":itn_backend,"metrics":metrics,"original_url":f"/api/jobs/{job_id}/audio/normalized","denoised_url":f"/api/jobs/{job_id}/audio/denoised"}
            self.update(job_id,status="complete",stage="complete",progress=100,result=result)
        except Exception as exc:
            self.update(job_id,status="failed",stage="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        finally:
            if biasing_applied:
                try:self.asr.reset_decoding()
                except Exception:pass  # never let cleanup mask the job result
    def _configure_biasing(self,language,biasing_context,job_dir):
        """Build a per-row boosting-tree phrase pack (Hindi only) and switch the ASR
        decode to bias toward the row's name/lender/amount/date. Returns a metrics
        dict; on any failure the job continues unbiased rather than erroring."""
        if not biasing_context:return None
        from biasing_context import build_key_phrases_file,is_hindi
        if not is_hindi(language):return {"applied":False,"reason":"language_not_hindi"}
        try:
            built=build_key_phrases_file(biasing_context,language=language,out_dir=job_dir/"biasing")
            if not built:return {"applied":False,"reason":"no_usable_fields"}
            key_file,pack=built
            self.asr.configure_biasing(key_file)
            return {"applied":True,"dynamic_phrases":pack.phrase_count_after_pruning,"total_phrases":pack.total_phrase_count,"top_phrases":list(pack.top_phrases),"key_phrases_file":str(key_file)}
        except Exception as exc:
            return {"applied":False,"error":f"{type(exc).__name__}: {exc}"}
    def audio_path(self,job_id,kind):
        if kind not in {"normalized","denoised"}:return None
        path=self.root/job_id/f"{kind}.wav"; return path if path.exists() else None
    def cleanup(self):
        cutoff=time.time()-self.ttl
        with self.lock: expired=[job_id for job_id,job in self.jobs.items() if job["created_at"]<cutoff]
        for job_id in expired:
            shutil.rmtree(self.root/job_id,ignore_errors=True)
            with self.lock:self.jobs.pop(job_id,None)


def _normalize_turn(text: str, language: str, backend: str) -> dict[str, Any]:
    """Run final-only offline ITN without ever blocking transcript delivery."""
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
