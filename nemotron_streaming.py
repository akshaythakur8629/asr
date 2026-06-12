"""Cache-aware adapter for nvidia/nemotron-3.5-asr-streaming-0.6b."""
from __future__ import annotations
import threading
import asyncio
import time
from typing import NamedTuple, List
from pathlib import Path

class TranscriptionRequest(NamedTuple):
    audio_path: Path
    language: str
    future: asyncio.Future

MODEL_NAME="nvidia/nemotron-3.5-asr-streaming-0.6b"
DECODING_STRATEGY="maes"  # RNN-T beam search; higher accuracy than greedy when latency is irrelevant
BEAM_SIZE=4
FULL_CONTEXT=[-1,-1]  # unlimited left+right attention = offline full-context decoding, the most accurate setting
ATTENTION_CONTEXTS={80:[56,0],160:[56,1],320:[56,3],560:[56,6],1120:[56,13]}
# NeMo's GPU boosting tree only attaches to *batched* beam strategies (it raises
# NotImplementedError for `maes`), so biased turns decode with malsd_batch.
BIASING_STRATEGY="malsd_batch"
BIASING_ALPHA=1.0       # fusion weight of the boosting tree against the acoustic/LM score
BIASING_CONTEXT_SCORE=1.5  # per-arc boost inside the context graph; >~2 makes the decoder
                           # regurgitate boosted phrases regardless of the audio (runaway)

def _prompt_language(language: str | None) -> str:
    value=str(language or "hi-IN").strip()
    if not value:return "hi-IN"
    lowered=value.lower().replace("_", "-")
    if lowered in {"hi", "hindi", "hin", "hi-in", "hi in", "हिंदी", "हिन्दी"}:return "hi-IN"
    if lowered.split("-",1)[0] == "hi":return "hi-IN"
    return value

def _prompt_transcribe_config(model, prompt_lang: str, *, batch_size: int, verbose: bool):
    """Build a prompt-model transcribe config that avoids NeMo's Lhotse prompt-index inference path."""
    get_config=getattr(model,"get_transcribe_config",None)
    if not callable(get_config):return None
    cfg=get_config()
    if not hasattr(cfg,"target_lang"):return None
    if hasattr(cfg,"use_lhotse"):cfg.use_lhotse=False
    cfg.target_lang=prompt_lang
    if hasattr(cfg,"batch_size"):cfg.batch_size=batch_size
    if hasattr(cfg,"verbose"):cfg.verbose=verbose
    if hasattr(cfg,"num_workers"):cfg.num_workers=0
    return cfg

def _hypothesis_text(item) -> str:
    """Extract the decoded string from one clip's beam-search result.

    With beam decoding (``maes``/``malsd_batch``) NeMo returns the n-best set per
    utterance — an ``NBestHypotheses`` wrapper, or a plain ``list``/``tuple`` of
    ``Hypothesis`` — neither of which has a ``.text`` attribute. The old code did
    ``str(getattr(item,"text",item))`` and so dumped the whole ``[Hypothesis(...), ...]``
    repr (100k+ chars of tensors and scores) into the transcript. Unwrap to the single
    best hypothesis's text, returning "" rather than an object repr on anything unexpected.
    """
    nbest=getattr(item,"n_best_hypotheses",None)
    if nbest:item=nbest
    if isinstance(item,(list,tuple)):item=item[0] if item else ""
    text=getattr(item,"text",item)
    return str(text).strip() if text is not None else ""

class NemotronStreamingASR:
    def __init__(self, device="cuda:0"):
        import torch
        from nemo.collections.asr.models import ASRModel
        self.torch=torch; self.device=torch.device(device); self.lock=threading.Lock()
        self.model=ASRModel.from_pretrained(MODEL_NAME,map_location=self.device).to(device=self.device,dtype=torch.float32)
        decoding_cfg=self.model.cfg.decoding.copy(); decoding_cfg.strategy=DECODING_STRATEGY
        if "beam" in decoding_cfg: decoding_cfg.beam.beam_size=BEAM_SIZE
        self.model.change_decoding_strategy(decoding_cfg,verbose=False); self.model.eval()
        self._base_decoding_cfg=decoding_cfg  # restored by reset_decoding() after a biased job
        self._biasing_active=False
        
        # Async batching queue fields
        self.queue = None
        self.loop = None
        self.batch_worker_task = None
    def configure_biasing(self,key_phrases_file,*,source_lang="hi",alpha=BIASING_ALPHA,context_score=BIASING_CONTEXT_SCORE,use_triton=None)->None:
        """Switch the decode to a boosting-tree-aware batched strategy for subsequent
        transcribe() calls. Per-job (one debtor): call once before the turn loop and
        reset_decoding() in a finally. Raises if this NeMo build / model can't bias a
        transducer (no batched strategy, no tokenizer, etc.) — callers should fall back."""
        import dataclasses
        from omegaconf import OmegaConf
        from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig
        bt=BoostingTreeModelConfig(key_phrases_file=str(key_phrases_file),source_lang=source_lang,context_score=context_score)
        if use_triton is not None: bt.use_triton=use_triton
        with self.lock:
            self.torch.cuda.set_device(self.device)
            cfg=self.model.cfg.decoding.copy(); OmegaConf.set_struct(cfg,False)
            cfg.strategy=BIASING_STRATEGY
            if "beam" not in cfg or cfg.beam is None: cfg.beam=OmegaConf.create({})
            cfg.beam.beam_size=BEAM_SIZE
            cfg.beam.boosting_tree=OmegaConf.create(dataclasses.asdict(bt))
            cfg.beam.boosting_tree_alpha=float(alpha)
            self.model.change_decoding_strategy(cfg,verbose=False)
            self._biasing_active=True
    def reset_decoding(self)->None:
        """Restore the default maes decode. Safe to call unconditionally."""
        if not getattr(self,"_biasing_active",False): return
        with self.lock:
            self.torch.cuda.set_device(self.device)
            self.model.change_decoding_strategy(self._base_decoding_cfg,verbose=False)
            self._biasing_active=False
    def transcribe(self,audio:Path,language="hi-IN",chunk_ms=None)->str:
        """Offline full-context transcription of a turn clip — the highest-accuracy path.

        Decodes the whole clip with unlimited right-context instead of emulating real-time
        streaming, which is strictly more accurate when latency does not matter. `chunk_ms`
        is accepted for API compatibility but ignored; use transcribe_streaming() for the
        real-time cache-aware path.
        """
        with self.lock:
            self.torch.cuda.set_device(self.device)
            self.model.encoder.set_default_att_context_size(att_context_size=FULL_CONTEXT)
            prompt_lang=_prompt_language(language)
            if hasattr(self.model,"set_inference_prompt"):
                self.model.set_inference_prompt(prompt_lang); self.model.decoding.set_strip_lang_tags(True)
            with self.torch.inference_mode():
                override_config=_prompt_transcribe_config(self.model,prompt_lang,batch_size=1,verbose=False)
                kwargs={"batch_size":1,"verbose":False,"target_lang":prompt_lang}
                if override_config is not None:kwargs["override_config"]=override_config
                results=self.model.transcribe([str(audio)],**kwargs)
        if isinstance(results,tuple): results=results[0]  # transducer .transcribe may return (best, all)
        return _hypothesis_text(results[0] if results else "")
    def transcribe_streaming(self,audio:Path,language="hi-IN",chunk_ms=1120)->str:
        """Real-time cache-aware streaming path (fallback). Defaults to the most accurate wired
        context [56,13] (1120 ms lookahead) rather than 320 ms."""
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        context=ATTENTION_CONTEXTS.get(chunk_ms)
        if context is None: raise ValueError(f"Unsupported chunk size: {chunk_ms}")
        with self.lock:
            self.torch.cuda.set_device(self.device)
            self.model.encoder.set_default_att_context_size(att_context_size=context)
            prompt_lang=_prompt_language(language)
            if hasattr(self.model,"set_inference_prompt"):
                self.model.set_inference_prompt(prompt_lang); self.model.decoding.set_strip_lang_tags(True)
            buffer=CacheAwareStreamingAudioBuffer(model=self.model,online_normalization=False); buffer.append_audio_file(str(audio),stream_id=-1)
            channel,time_cache,channel_len=self.model.encoder.get_initial_cache_state(batch_size=1); previous=None; outputs=None; texts=[]
            for step,(chunk,lengths) in enumerate(buffer):
                with self.torch.inference_mode():
                    outputs,texts,channel,time_cache,channel_len,previous=self.model.conformer_stream_step(processed_signal=chunk.to(device=self.device, dtype=self.torch.float32),processed_signal_length=lengths.to(self.device),cache_last_channel=channel,cache_last_time=time_cache,cache_last_channel_len=channel_len,keep_all_outputs=buffer.is_buffer_empty(),previous_hypotheses=previous,previous_pred_out=outputs,drop_extra_pre_encoded=0 if step==0 else self.model.encoder.streaming_cfg.drop_extra_pre_encoded,return_transcription=True)
            if not texts:return ""
            return _hypothesis_text(texts[0])

    async def transcribe_async(self, audio: Path, language="hi-IN") -> str:
        # Fallback to standard transcribe if biasing is active (due to an offline batch job running)
        if self._biasing_active:
            return await asyncio.to_thread(self.transcribe, audio, language)
            
        if self.queue is None:
            self.queue = asyncio.Queue()
            self.loop = asyncio.get_event_loop()
            self.batch_worker_task = asyncio.create_task(self._batch_worker())
            
        future = self.loop.create_future()
        await self.queue.put(TranscriptionRequest(audio_path=audio, language=language, future=future))
        return await future

    async def _batch_worker(self):
        import os
        max_batch_size = int(os.environ.get("ASR_MAX_BATCH_SIZE", "16"))
        batch_timeout_sec = float(os.environ.get("ASR_BATCH_TIMEOUT_SEC", "0.05"))
        
        while True:
            try:
                # Wait for the first item in the batch
                req = await self.queue.get()
                batch = [req]
                
                # Dynamic batching collection window
                start_time = time.time()
                while len(batch) < max_batch_size:
                    try:
                        # Non-blocking drain from queue
                        req_next = self.queue.get_nowait()
                        batch.append(req_next)
                    except asyncio.QueueEmpty:
                        elapsed = time.time() - start_time
                        remaining = batch_timeout_sec - elapsed
                        if remaining <= 0:
                            break
                        try:
                            req_next = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                            batch.append(req_next)
                        except asyncio.TimeoutError:
                            break
                
                # Group request batch by target language
                groups = {}
                for r in batch:
                    groups.setdefault(r.language, []).append(r)
                    
                for lang, group_reqs in groups.items():
                    audio_paths = [str(r.audio_path) for r in group_reqs]
                    try:
                        # Process blocking batch inference in a separate thread
                        results = await asyncio.to_thread(self._transcribe_batch, audio_paths, lang)
                        for r, text in zip(group_reqs, results):
                            if not r.future.done():
                                r.future.set_result(text)
                    except Exception as e:
                        for r in group_reqs:
                            if not r.future.done():
                                r.future.set_exception(e)
                    finally:
                        for _ in range(len(group_reqs)):
                            self.queue.task_done()
            except Exception as e:
                # Catch worker errors to keep loop alive
                print(f"ASR batch worker exception: {e}")
                await asyncio.sleep(0.1)

    def _transcribe_batch(self, audio_paths: List[str], language: str) -> List[str]:
        with self.lock:
            self.torch.cuda.set_device(self.device)
            self.model.encoder.set_default_att_context_size(att_context_size=FULL_CONTEXT)
            prompt_lang=_prompt_language(language)
            if hasattr(self.model, "set_inference_prompt"):
                self.model.set_inference_prompt(prompt_lang)
                self.model.decoding.set_strip_lang_tags(True)
            with self.torch.inference_mode():
                override_config=_prompt_transcribe_config(self.model, prompt_lang, batch_size=len(audio_paths), verbose=False)
                kwargs = {"batch_size": len(audio_paths), "verbose": False, "target_lang": prompt_lang}
                if override_config is not None:
                    kwargs["override_config"] = override_config
                results = self.model.transcribe(audio_paths, **kwargs)
        if isinstance(results, tuple):
            results = results[0]
        return [_hypothesis_text(res) for res in results]