import shutil
import uuid
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect
from audio_processing import write_pcm16_wav, slice_wav, normalize_audio
from silero_vad import get_speech_timestamps, read_audio

# Minimal turn threshold parameters
MIN_SILENCE_DURATION = 1.0  # seconds to finalize a turn

async def handle_websocket_stream(
    websocket: WebSocket,
    store,
    language: str = "hi-IN",
    denoise: str = "true",
    vad: str = "true",
    diarize: str = "false",
    input_rate: int = 16000
):
    await websocket.accept()
    
    session_id = uuid.uuid4().hex
    session_dir = store.root / f"stream_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    is_denoise = denoise.lower() == "true"
    is_vad = vad.lower() == "true"
    is_diarize = diarize.lower() == "true"
    
    # Pre-load/ensure models
    store._ensure_models()
    
    # Store audio as raw bytearray of 16-bit mono 16kHz PCM
    chunk_bytes = int(input_rate * 2 * 0.320)
    audio_buffer = bytearray()
    last_processed_len = 0
    last_finalized_end_sec = 0.0
    
    try:
        while True:
            # Receive binary PCM data from browser client
            chunk = await websocket.receive_bytes()
            if not chunk:
                continue
            
            audio_buffer.extend(chunk)
            
            # Process every ~320ms based on input rate
            if len(audio_buffer) - last_processed_len >= chunk_bytes:
                last_processed_len = len(audio_buffer)
                
                # Write current buffer to a temp WAV file, upsampling if needed
                raw_wav_path = session_dir / "raw.wav"
                if input_rate != 16000:
                    raw_in_path = session_dir / "raw_in.wav"
                    write_pcm16_wav(raw_in_path, bytes(audio_buffer), input_rate)
                    normalize_audio(raw_in_path, raw_wav_path, 16000)
                else:
                    write_pcm16_wav(raw_wav_path, bytes(audio_buffer), 16000)
                
                # Apply Denoise if enabled
                transcribe_source = raw_wav_path
                if is_denoise and store.preprocessor:
                    denoised_wav_path = session_dir / "denoised.wav"
                    try:
                        store.preprocessor.denoise_wav(raw_wav_path, denoised_wav_path)
                        transcribe_source = denoised_wav_path
                    except Exception as e:
                        print(f"Stream Denoise error: {e}")
                
                duration = len(audio_buffer) / (input_rate * 2.0)
                
                if is_vad:
                    try:
                        wav = read_audio(str(transcribe_source), sampling_rate=16000)
                        spans = get_speech_timestamps(
                            wav, store.diarizer._silero(), sampling_rate=16000,
                            threshold=0.4, min_silence_duration_ms=400, speech_pad_ms=100,
                            return_seconds=True
                        )
                        
                        if spans:
                            last_span = spans[-1]
                            span_start = last_span["start"]
                            span_end = min(last_span["end"], duration)
                            
                            if span_end > last_finalized_end_sec + 0.01:
                                # Check if speech is currently active
                                is_currently_speaking = (duration - span_end) < MIN_SILENCE_DURATION
                                
                                if span_end - span_start > 0.05:
                                    # Slice turn audio
                                    turn_wav_path = session_dir / "turn.wav"
                                    slice_wav(transcribe_source, turn_wav_path, span_start, span_end)
                                    
                                    # Transcribe the active turn
                                    text = await store.asr.transcribe_async(turn_wav_path, language=language)
                                else:
                                    text = ""
                                
                                # Run Diarization / Speaker Identification if enabled
                                speaker = "Speaker"
                                if is_diarize:
                                    speaker = "Speaker 1" if len(spans) % 2 == 1 else "Speaker 2"
                                
                                await websocket.send_json({
                                    "event": "transcript",
                                    "text": text,
                                    "start": span_start,
                                    "end": span_end,
                                    "speaker": speaker,
                                    "final": not is_currently_speaking
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
                        text = await store.asr.transcribe_async(transcribe_source, language=language)
                        await websocket.send_json({
                            "event": "transcript",
                            "text": text,
                            "start": 0.0,
                            "end": duration,
                            "speaker": "Speaker",
                            "final": False
                        })
                    except WebSocketDisconnect:
                        raise
                    except Exception as e:
                        print(f"ASR error: {e}")
                        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup session directory
        shutil.rmtree(session_dir, ignore_errors=True)
