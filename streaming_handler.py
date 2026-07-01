import logging
from logging.handlers import RotatingFileHandler
import shutil
import uuid
import asyncio
import time
import json
import base64
from datetime import datetime
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect
from utils.audio_processing import write_pcm16_wav, slice_wav, normalize_audio, read_pcm16_wav
from silero_vad import get_speech_timestamps, read_audio
from utils.pipeline import resolve_model

# Setup a clean logger that writes ONLY request and response events to app.log
req_res_logger = logging.getLogger("req_res_logger")
req_res_logger.setLevel(logging.INFO)
req_res_logger.handlers = []
req_res_logger.propagate = False

try:
    log_dir = Path("/app/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    f_handler = RotatingFileHandler(log_dir / "app.log", maxBytes=10*1024*1024, backupCount=5)
    f_handler.setFormatter(logging.Formatter('%(message)s'))
    req_res_logger.addHandler(f_handler)
except Exception as err:
    print(f"Error initializing app.log file logger: {err}")

def log_event(msg: str):
    print(msg, flush=True)
    try:
        req_res_logger.info(msg)
    except Exception:
        pass

# Minimal turn threshold parameters
MIN_SILENCE_DURATION = 1.0  # seconds to finalize a turn

async def handle_websocket_stream(
    websocket: WebSocket,
    store,
    language: str = "hi-IN",
    denoise: str = "true",
    vad: str = "true",
    diarize: str = "false",
    input_rate: int = 16000,
    chunk_ms: int = 160,
    call_code: str = None,
    flush_signal: str = "false",
    model: str = "sarga:v1",
    itn: str = "false",
    itn_backend: str = "custom"
):
    await websocket.accept()
    model = resolve_model(model)
    
    session_id = call_code if call_code else uuid.uuid4().hex
    
    async def send_json_logged(payload: dict):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_event(f"[{now_str}] [WS] [RES] [Session: {session_id}] Sending: {json.dumps(payload, ensure_ascii=False)}")
        await websocket.send_json(payload)
 
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log_event(f"[{now_str}] [WS] [OPEN] Session {session_id} connected. chunk_ms={chunk_ms}, language={language}, denoise={denoise}, vad={vad}, flush_signal={flush_signal}, model={model}, itn={itn}, itn_backend={itn_backend}")
    session_dir = store.root / f"stream_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    is_denoise = denoise.lower() == "true"
    is_vad = vad.lower() == "true"
    is_diarize = diarize.lower() == "true"
    is_flush = flush_signal.lower() == "true"
    is_itn = itn.lower() == "true"
    is_auto_lid = (language == "auto")
    
    async def _detect_language_if_auto(wav_path: Path):
        nonlocal language
        if not is_auto_lid:
            return language
        
        if store.lid_service is not None:
            try:
                pcm, sr = await asyncio.to_thread(read_pcm16_wav, wav_path)
                fallback_langs = {"as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"}
                supported = getattr(store.asr_indic, "supported_languages", fallback_langs) if store.asr_indic is not None else fallback_langs
                
                res = await asyncio.to_thread(store.lid_service.identify_turn_language, pcm, sr, supported)
                if res.language:
                    detected = res.language
                else:
                    detected = "hi"
                
                locale_map = {"hi": "hi-IN", "te": "te-IN", "ta": "ta-IN", "mr": "mr-IN"}
                language = locale_map.get(detected, f"{detected}-IN")
                log_event(f"[LID] Resolved 'auto' language to: {language} (confidence: {res.confidence:.2f})")
                
                # Send the detection event to client
                await send_json_logged({
                    "event": "language_detected",
                    "language_code": language.split("-", 1)[0],
                    "confidence": res.confidence
                })
            except Exception as e:
                log_event(f"[LID] Auto-LID detection failed: {e}. Defaulting to hi-IN.")
                language = "hi-IN"
        else:
            language = "hi-IN"
        return language

    # Pre-load/ensure models
    store._ensure_models()

    if model == "credresolve:v1" and store.asr_indic is None:
        log_event(f"[{now_str}] [WS] [CLOSE] Session {session_id} rejected. Model {model} (Indic) not deployed.")
        await websocket.close(code=4000, reason=f"Model {model} not deployed")
        return
    elif model == "sarga:v1" and store.asr is None:
        log_event(f"[{now_str}] [WS] [CLOSE] Session {session_id} rejected. Model sarga:v1 (Nemotron) not deployed.")
        await websocket.close(code=4000, reason="Model sarga:v1 not deployed")
        return

    def get_asr_worker():
        if model == "credresolve:v1":
            return store.asr_indic
        return store.asr
    
    # Store audio as raw bytearray of 16-bit mono 16kHz PCM
    audio_buffer = bytearray()
    flush_requested = False
    
    async def receive_loop():
        nonlocal last_data_time, flush_requested, last_processed_len, input_rate, model
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    raw_bytes = msg["bytes"]
                    log_event(f"[DEBUG_PACKET] binary input_rate={input_rate}, len(raw_bytes)={len(raw_bytes)}, hex={raw_bytes[:16].hex()}")
                    audio_buffer.extend(raw_bytes)
                    last_data_time = time.time()
                elif "text" in msg and msg["text"]:
                    try:
                        # Log text message structure to diagnose telco payload format
                        data = json.loads(msg["text"])
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        if isinstance(data, dict):
                            msg_type = data.get("type")
                            log_data = data.copy()
                            if "audio" in log_data and isinstance(log_data["audio"], dict):
                                audio_copy = log_data["audio"].copy()
                                if "data" in audio_copy:
                                    audio_copy["data"] = f"<base64: {len(audio_copy['data'])} chars>"
                                log_data["audio"] = audio_copy
                            log_event(f"[{now_str}] [WS] [REQ] [Session: {session_id}] Received: {json.dumps(log_data, ensure_ascii=False)}")
                            
                            # Support telco base64 audio payload
                            if "audio" in data and isinstance(data["audio"], dict):
                                audio_data = data["audio"].get("data")
                                s_rate = data["audio"].get("sample_rate")
                                if s_rate and str(s_rate).isdigit():
                                    input_rate = int(s_rate)
                                if audio_data:
                                    raw_bytes = base64.b64decode(audio_data)
                                    log_event(f"[DEBUG_PACKET] input_rate={input_rate}, len(raw_bytes)={len(raw_bytes)}, hex={raw_bytes[:16].hex()}")
                                    audio_buffer.extend(raw_bytes)
                                    last_data_time = time.time()
                            
                            # Support control frames
                            if msg_type == "flush":
                                flush_requested = True
                            elif msg_type == "start":
                                call_id = data.get("call_id")
                                req_model = data.get("model") or data.get("model_name")
                                if req_model:
                                    req_model = resolve_model(req_model)
                                    if req_model == "credresolve:v1" and store.asr_indic is None:
                                        await send_json_logged({"type": "error", "message": f"Model {req_model} not deployed"})
                                        break
                                    elif req_model == "sarga:v1" and store.asr is None:
                                        await send_json_logged({"type": "error", "message": f"Model {req_model} not deployed"})
                                        break
                                    model = req_model
                                audio_buffer.clear()
                                last_processed_len = 0
                                await send_json_logged({"type": "ready", "call_id": call_id})
                            elif msg_type == "stop":
                                await send_json_logged({"type": "done"})
                                break
                        else:
                            log_event(f"[{now_str}] [WS] [REQ] [Session: {session_id}] Received text: {msg['text'][:200]}")
                    except Exception as e:
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        log_event(f"[{now_str}] [WS] [DEBUG] Session {session_id} text parse error: {e}. Raw content: {msg['text'][:200]}")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_event(f"[{now_str}] [WS] [ERROR] Session {session_id} receive error: {e}")
            
    receive_task = asyncio.create_task(receive_loop())
    
    last_processed_len = 0
    last_finalized_end_sec = 0.0
    last_data_time = time.time()
    
    is_frontend = str(session_id).startswith("frontend-session")
    
    try:
        # Determine loop interval (e.g. chunk_ms / 1000.0, fallback to 160ms if 0)
        check_interval = (chunk_ms / 1000.0) if chunk_ms > 0 else 0.160
        
        while not receive_task.done():
            await asyncio.sleep(check_interval)
            
            # Non-frontend (Telco / Bot) session logic
            if not is_frontend:
                # Case A: If client requested explicit flush-gating (flush_signal=true)
                if is_flush:
                    if not flush_requested:
                        continue
                    
                    flush_requested = False
                    current_len = len(audio_buffer)
                    duration = current_len / (input_rate * 2.0)
                    
                    if current_len > 0:
                        raw_wav_path = session_dir / "raw.wav"
                        if input_rate != 16000:
                            raw_in_path = session_dir / "raw_in.wav"
                            await asyncio.to_thread(write_pcm16_wav, raw_in_path, bytes(audio_buffer), input_rate)
                            try:
                                import shutil
                                shutil.copyfile(raw_in_path, "/app/logs/debug.wav")
                                log_event(f"[DEBUG_WAV] Saved 8kHz raw input to /app/logs/debug.wav")
                            except Exception as ex:
                                log_event(f"[DEBUG_WAV] Error saving copy: {ex}")
                            await asyncio.to_thread(normalize_audio, raw_in_path, raw_wav_path, 16000)
                        else:
                            await asyncio.to_thread(write_pcm16_wav, raw_wav_path, bytes(audio_buffer), 16000)
                            try:
                                import shutil
                                shutil.copyfile(raw_wav_path, "/app/logs/debug.wav")
                                log_event(f"[DEBUG_WAV] Saved 16kHz raw input to /app/logs/debug.wav")
                            except Exception as ex:
                                log_event(f"[DEBUG_WAV] Error saving copy: {ex}")
                        
                        transcribe_source = raw_wav_path
                        if is_denoise and store.preprocessor:
                            denoised_wav_path = session_dir / "denoised.wav"
                            try:
                                await asyncio.to_thread(store.preprocessor.denoise_wav, raw_wav_path, denoised_wav_path)
                                transcribe_source = denoised_wav_path
                            except Exception as e:
                                print(f"Flush Denoise error: {e}")
                        
                        try:
                            if is_auto_lid:
                                await _detect_language_if_auto(transcribe_source)
                            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting full audio: {duration:.2f}s on flush, Language: {language}")
                            
                            t_start = time.time()
                            text = await get_asr_worker().transcribe_async(transcribe_source, language=language)
                            t_end = time.time()
                            inference_time_ms = (t_end - t_start) * 1000.0
                            
                            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            log_event(f"[{now_str}] [ASR] [RES] [Session: {session_id}] Received text: '{text}' (Inference Time: {inference_time_ms:.1f}ms, Language: {language})")
                            
                            if is_itn and text.strip():
                                from utils.pipeline import _normalize_turn
                                text = _normalize_turn(text, language, itn_backend).get("canonical_text", text)
                            
                            response_data = {
                                "type": "data",
                                "data": {
                                    "transcript": text,
                                    "language_code": language.split("-", 1)[0],
                                    "metrics": {
                                        "inference_time": round(inference_time_ms, 2)
                                    }
                                }
                            }
                            await send_json_logged(response_data)
                        except Exception as e:
                            print(f"ASR error on flush: {e}")
                    else:
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting empty buffer (0.00s) on flush")
                        
                        response_data = {
                            "type": "data",
                            "data": {
                                "transcript": "",
                                "language_code": language.split("-", 1)[0],
                                "metrics": {
                                    "inference_time": 0.0
                                }
                            }
                        }
                        await send_json_logged(response_data)
                    
                    audio_buffer.clear()
                    last_processed_len = 0
                    last_finalized_end_sec = 0.0
                    continue
                
                # Case B: If client did NOT request explicit flush-gating (flush_signal=false),
                # we run ASR and flush the audio buffer only after 350ms of silence/inactivity.
                else:
                    current_len = len(audio_buffer)
                    if current_len > 0 and (time.time() - last_data_time) > 0.350:
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        log_event(f"[{now_str}] [WS] [FLUSH] Session {session_id} auto-flushing due to 350ms inactivity.")
                        duration = current_len / (input_rate * 2.0)
                        
                        raw_wav_path = session_dir / "raw.wav"
                        if input_rate != 16000:
                            raw_in_path = session_dir / "raw_in.wav"
                            await asyncio.to_thread(write_pcm16_wav, raw_in_path, bytes(audio_buffer), input_rate)
                            await asyncio.to_thread(normalize_audio, raw_in_path, raw_wav_path, 16000)
                        else:
                            await asyncio.to_thread(write_pcm16_wav, raw_wav_path, bytes(audio_buffer), 16000)
                        
                        transcribe_source = raw_wav_path
                        if is_denoise and store.preprocessor:
                            denoised_wav_path = session_dir / "denoised.wav"
                            try:
                                await asyncio.to_thread(store.preprocessor.denoise_wav, raw_wav_path, denoised_wav_path)
                                transcribe_source = denoised_wav_path
                            except Exception as e:
                                print(f"Auto-Flush Denoise error: {e}")
                        
                        try:
                            if language == "auto":
                                await _detect_language_if_auto(transcribe_source)
                            log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting full audio: {duration:.2f}s on auto-flush")
                            t_start = time.time()
                            text = await get_asr_worker().transcribe_async(transcribe_source, language=language)
                            t_end = time.time()
                            inference_time_ms = (t_end - t_start) * 1000.0
                            
                            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            log_event(f"[{now_str}] [ASR] [RES] [Session: {session_id}] Received text: '{text}' (Inference Time: {inference_time_ms:.1f}ms)")
                            
                            response_data = {
                                "type": "data",
                                "data": {
                                    "transcript": text,
                                    "language_code": language,
                                    "metrics": {
                                        "inference_time": round(inference_time_ms, 2)
                                    }
                                }
                            }
                            await send_json_logged(response_data)
                        except Exception as e:
                            print(f"ASR error on auto-flush: {e}")
                        
                        audio_buffer.clear()
                        last_processed_len = 0
                        last_finalized_end_sec = 0.0
                    continue
            
            current_len = len(audio_buffer)
            if current_len == last_processed_len:
                # Auto-flush on inactivity: if audio buffer exists and we haven't received data for 1.0s
                if current_len > 0 and (time.time() - last_data_time) > 1.0:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    log_event(f"[{now_str}] [WS] [FLUSH] Session {session_id} auto-flushed due to 1.0s inactivity.")
                    audio_buffer.clear()
                    last_processed_len = 0
                    last_finalized_end_sec = 0.0
                continue
                
            last_processed_len = current_len
            
            raw_wav_path = session_dir / "raw.wav"
            if input_rate != 16000:
                raw_in_path = session_dir / "raw_in.wav"
                await asyncio.to_thread(write_pcm16_wav, raw_in_path, bytes(audio_buffer), input_rate)
                await asyncio.to_thread(normalize_audio, raw_in_path, raw_wav_path, 16000)
            else:
                await asyncio.to_thread(write_pcm16_wav, raw_wav_path, bytes(audio_buffer), 16000)
            
            # Apply Denoise if enabled
            transcribe_source = raw_wav_path
            if is_denoise and store.preprocessor:
                denoised_wav_path = session_dir / "denoised.wav"
                try:
                    await asyncio.to_thread(store.preprocessor.denoise_wav, raw_wav_path, denoised_wav_path)
                    transcribe_source = denoised_wav_path
                except Exception as e:
                    print(f"Stream Denoise error: {e}")
            
            duration = len(audio_buffer) / (input_rate * 2.0)
            
            if is_vad:
                try:
                    # Only run VAD on the audio segment after the last finalized turn
                    # This avoids O(N^2) CPU and IO scaling with call duration
                    if duration - last_finalized_end_sec < 0.1:
                        continue
                        
                    vad_wav_path = session_dir / "vad_slice.wav"
                    await asyncio.to_thread(slice_wav, transcribe_source, vad_wav_path, last_finalized_end_sec, duration)
                    
                    wav = await asyncio.to_thread(read_audio, str(vad_wav_path), sampling_rate=16000)
                    spans = await asyncio.to_thread(
                        get_speech_timestamps,
                        wav, store._silero(), sampling_rate=16000,
                        threshold=0.4, min_silence_duration_ms=400, speech_pad_ms=100,
                        return_seconds=True
                    )
                    
                    if spans:
                        last_span = spans[-1]
                        span_start = last_span["start"] + last_finalized_end_sec
                        span_end = min(last_span["end"] + last_finalized_end_sec, duration)
                        
                        if span_end > last_finalized_end_sec + 0.01:
                            # Check if speech is currently active
                            is_currently_speaking = (duration - span_end) < MIN_SILENCE_DURATION
                            
                            if span_end - span_start > 0.05:
                                # Slice turn audio
                                turn_wav_path = session_dir / "turn.wav"
                                await asyncio.to_thread(slice_wav, transcribe_source, turn_wav_path, span_start, span_end)
                                
                                # Transcribe the active turn
                                if is_auto_lid:
                                    await _detect_language_if_auto(turn_wav_path)
                                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting turn slice: {span_start:.2f}s - {span_end:.2f}s, Language: {language}")
                                t_start = time.time()
                                text = await get_asr_worker().transcribe_async(turn_wav_path, language=language)
                                t_end = time.time()
                                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                log_event(f"[{now_str}] [ASR] [RES] [Session: {session_id}] Received text: '{text}' (Inference Time: {(t_end - t_start)*1000:.1f}ms, Language: {language})")
                                if is_itn and text.strip():
                                    from utils.pipeline import _normalize_turn
                                    text = _normalize_turn(text, language, itn_backend).get("canonical_text", text)
                            else:
                                text = ""
                            
                            # Run Diarization / Speaker Identification if enabled
                            speaker = "Speaker"
                            if is_diarize:
                                speaker = "Speaker 1" if len(spans) % 2 == 1 else "Speaker 2"
                            
                            is_final = not is_currently_speaking
                            if is_frontend:
                                await send_json_logged({
                                    "event": "transcript",
                                    "type": "final" if is_final else "partial",
                                    "text": text,
                                    "start": span_start,
                                    "end": span_end,
                                    "speaker": speaker,
                                    "final": is_final,
                                    "diarize": is_diarize,
                                    "language_code": language.split("-", 1)[0],
                                    "ts_ms": int(time.time() * 1000)
                                })
                            else:
                                t_diff = (t_end - t_start) * 1000.0 if 't_start' in locals() and 't_end' in locals() else 0.0
                                await send_json_logged({
                                    "type": "data",
                                    "data": {
                                        "transcript": text,
                                        "language_code": language.split("-", 1)[0],
                                        "metrics": {
                                            "inference_time": round(t_diff, 2)
                                        }
                                    }
                                })
                            
                            if not is_currently_speaking:
                                last_finalized_end_sec = span_end
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"VAD error in websocket: {e}")
            else:
                # Direct transcript mode: transcribe the entire accumulated audio
                try:
                    if is_auto_lid:
                        await _detect_language_if_auto(transcribe_source)
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting full audio: {duration:.2f}s, Language: {language}")
                    t_start = time.time()
                    text = await get_asr_worker().transcribe_async(transcribe_source, language=language)
                    t_end = time.time()
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    log_event(f"[{now_str}] [ASR] [RES] [Session: {session_id}] Received text: '{text}' (Inference Time: {(t_end - t_start)*1000:.1f}ms, Language: {language})")
                    if is_itn and text.strip():
                        from utils.pipeline import _normalize_turn
                        text = _normalize_turn(text, language, itn_backend).get("canonical_text", text)
                    
                    if is_frontend:
                        await send_json_logged({
                            "event": "transcript",
                            "type": "partial",
                            "text": text,
                            "start": 0.0,
                            "end": duration,
                            "speaker": "Speaker",
                            "final": False,
                            "diarize": is_diarize,
                            "language_code": language.split("-", 1)[0],
                            "ts_ms": int(time.time() * 1000)
                        })
                    else:
                        t_diff = (t_end - t_start) * 1000.0 if 't_start' in locals() and 't_end' in locals() else 0.0
                        await send_json_logged({
                            "type": "data",
                            "data": {
                                "transcript": text,
                                "language_code": language.split("-", 1)[0],
                                "metrics": {
                                    "inference_time": round(t_diff, 2)
                                }
                            }
                        })
                    if text.strip() or is_flush:
                        audio_buffer.clear()
                        last_processed_len = 0
                        last_finalized_end_sec = 0.0
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    print(f"ASR error: {e}")
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_event(f"[{now_str}] [WS] [ERROR] Session {session_id} error: {e}")
    finally:
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_event(f"[{now_str}] [WS] [CLOSE] Session {session_id} disconnected.")
        try:
            if is_frontend:
                final_duration = len(audio_buffer) / (input_rate * 2.0)
                await send_json_logged({
                    "event": "transcript",
                    "type": "done",
                    "text": "",
                    "start": 0.0,
                    "end": final_duration,
                    "speaker": "Speaker",
                    "final": True,
                    "diarize": is_diarize,
                    "language_code": language.split("-", 1)[0],
                    "ts_ms": int(time.time() * 1000)
                })
        except Exception:
            pass

        # Cancel receive task to release WebSocket
        receive_task.cancel()
        try:
            await receive_task
        except Exception:
            pass
        # Cleanup session directory
        await asyncio.to_thread(shutil.rmtree, session_dir, ignore_errors=True)