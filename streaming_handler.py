import logging
from logging.handlers import RotatingFileHandler
import shutil
import uuid
import asyncio
import time
import json
from datetime import datetime
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect
from audio_processing import write_pcm16_wav, slice_wav, normalize_audio
from silero_vad import get_speech_timestamps, read_audio

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
    flush_signal: str = "false"
):
    await websocket.accept()
    
    session_id = call_code if call_code else uuid.uuid4().hex
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log_event(f"[{now_str}] [WS] [OPEN] Session {session_id} connected. chunk_ms={chunk_ms}, language={language}, denoise={denoise}, vad={vad}")
    session_dir = store.root / f"stream_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    is_denoise = denoise.lower() == "true"
    is_vad = vad.lower() == "true"
    is_diarize = diarize.lower() == "true"
    is_flush = flush_signal.lower() == "true"
    
    # Pre-load/ensure models
    store._ensure_models()
    
    # Store audio as raw bytearray of 16-bit mono 16kHz PCM
    audio_buffer = bytearray()
    
    async def receive_loop():
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"]:
                    audio_buffer.extend(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "start":
                            call_id = data.get("call_id")
                            audio_buffer.clear()
                            nonlocal last_processed_len
                            last_processed_len = 0
                            await websocket.send_json({"type": "ready", "call_id": call_id})
                        elif data.get("type") == "stop":
                            await websocket.send_json({"type": "done"})
                            break
                    except Exception as e:
                        print(f"Error parsing websocket text message: {e}")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_event(f"[{now_str}] [WS] [ERROR] Session {session_id} receive error: {e}")
            
    receive_task = asyncio.create_task(receive_loop())
    
    last_processed_len = 0
    last_finalized_end_sec = 0.0
    
    try:
        # Determine loop interval (e.g. chunk_ms / 1000.0, fallback to 160ms if 0)
        check_interval = (chunk_ms / 1000.0) if chunk_ms > 0 else 0.160
        
        while not receive_task.done():
            await asyncio.sleep(check_interval)
            
            current_len = len(audio_buffer)
            if current_len == last_processed_len:
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
                        wav, store.diarizer._silero(), sampling_rate=16000,
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
                                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting turn slice: {span_start:.2f}s - {span_end:.2f}s")
                                t_start = time.time()
                                text = await store.asr.transcribe_async(turn_wav_path, language=language)
                                t_end = time.time()
                                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                log_event(f"[{now_str}] [ASR] [RES] [Session: {session_id}] Received text: '{text}' (Inference Time: {(t_end - t_start)*1000:.1f}ms)")
                            else:
                                text = ""
                            
                            # Run Diarization / Speaker Identification if enabled
                            speaker = "Speaker"
                            if is_diarize:
                                speaker = "Speaker 1" if len(spans) % 2 == 1 else "Speaker 2"
                            
                            is_final = not is_currently_speaking
                            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            log_event(f"[{now_str}] [WS] [SEND] [Session: {session_id}] Sending transcript: '{text}' (start={span_start:.2f}s, end={span_end:.2f}s, final={is_final})")
                            
                            await websocket.send_json({
                                "event": "transcript",
                                "type": "final" if is_final else "partial",
                                "text": text,
                                "start": span_start,
                                "end": span_end,
                                "speaker": speaker,
                                "final": is_final,
                                "ts_ms": int(time.time() * 1000)
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
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    log_event(f"[{now_str}] [ASR] [REQ] [Session: {session_id}] Submitting full audio: {duration:.2f}s")
                    t_start = time.time()
                    text = await store.asr.transcribe_async(transcribe_source, language=language)
                    t_end = time.time()
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    log_event(f"[{now_str}] [ASR] [RES] [Session: {session_id}] Received text: '{text}' (Inference Time: {(t_end - t_start)*1000:.1f}ms)")
                    
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    log_event(f"[{now_str}] [WS] [SEND] [Session: {session_id}] Sending transcript: '{text}'")
                    await websocket.send_json({
                        "event": "transcript",
                        "type": "partial",
                        "text": text,
                        "start": 0.0,
                        "end": duration,
                        "speaker": "Speaker",
                        "final": False,
                        "ts_ms": int(time.time() * 1000)
                    })
                    if is_flush:
                        audio_buffer.clear()
                        last_processed_len = 0
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
            final_duration = len(audio_buffer) / (input_rate * 2.0)
            await websocket.send_json({
                "event": "transcript",
                "type": "done",
                "text": "",
                "start": 0.0,
                "end": final_duration,
                "speaker": "Speaker",
                "final": True,
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
