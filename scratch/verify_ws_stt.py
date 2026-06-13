import asyncio
import json
import time
import subprocess
import wave
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Error: websockets library is required.")
    sys.exit(1)

MP3_SOURCE = "/app/recording/1780486162178_audio_file_13161338920.mp3"
WAV_PATH = "/tmp/verify_stt.wav"

async def main():
    # 1. Normalize/resample MP3 to WAV inside the container
    print("Resampling sample MP3 to 16kHz mono WAV inside the container...")
    subprocess.run([
        "docker", "exec", "sarga-app", "ffmpeg", "-y", "-v", "error",
        "-i", MP3_SOURCE, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", WAV_PATH
    ], check=True)
    
    # Copy WAV from container to host to read it
    print("Copying WAV from container to host...")
    subprocess.run([
        "docker", "cp", f"sarga-app:{WAV_PATH}", "/tmp/verify_stt_host.wav"
    ], check=True)
    
    # 2. Read the audio frames
    print("Reading WAV audio data...")
    with wave.open("/tmp/verify_stt_host.wav", "rb") as f:
        pcm_data = f.readframes(f.getnframes())
        
    duration = len(pcm_data) / 32000.0
    print(f"Loaded {duration:.2f}s of audio data ({len(pcm_data)} bytes).")
    
    # Connect to the new /ws/stt endpoint with the same query parameters and headers
    uri = "ws://localhost:8000/ws/stt?language-code=hi&model=credresolve:v1&mode=transcribe&sample_rate=16000&vad_signals=false&flush_signal=true&input_audio_codec=pcm_s16le&binary_audio=1&call_code=my-custom-call-code"
    headers = {"Api-Subscription-Key": "dev"}
    print(f"Connecting to {uri} with headers {headers}...")
    
    # Send 64ms chunks (2048 bytes) every 64ms
    chunk_size = 2048
    num_chunks = len(pcm_data) // chunk_size
    send_times = {}
    all_latencies = []
    
    async with websockets.connect(uri, additional_headers=headers) as ws:
        async def receive_loop():
            try:
                async for msg in ws:
                    rec_time = time.time()
                    data = json.loads(msg)
                    print(f"Received JSON: {data}")
                    if data.get("event") == "transcript" or data.get("type") in ["partial", "final", "done"]:
                        text = data.get("text", "").strip()
                        span_start = data.get("start", 0.0)
                        span_end = data.get("end", 0.0)
                        is_final = data.get("final", False)
                        msg_type = data.get("type", "unknown")
                        
                        if text:
                            # Estimate the chunk index corresponding to span_end
                            idx = int(round(span_end * 16000 / (chunk_size / 2))) - 1
                            idx = max(0, min(idx, num_chunks - 1))
                            
                            if idx in send_times:
                                latency = rec_time - send_times[idx]
                                all_latencies.append(latency)
                                print(f"[{span_start:.2f}s - {span_end:.2f}s] Latency: {latency*1000:.1f}ms | Type: {msg_type} | Final: {is_final} | Text: {text}")
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"Receive loop error: {e}")

        recv_task = asyncio.create_task(receive_loop())
        
        print(f"Streaming {num_chunks} chunks of 64ms audio in real time...")
        start_stream_time = time.time()
        
        for i in range(num_chunks):
            # Calculate when this chunk should be sent to emulate real-time stream
            target_send_time = start_stream_time + (i * 0.064)
            sleep_time = target_send_time - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
            chunk = pcm_data[i * chunk_size : (i + 1) * chunk_size]
            send_times[i] = time.time()
            await ws.send(chunk)
            
        # Keep connection open briefly to get final responses
        await asyncio.sleep(3.0)
        await ws.close()
        await recv_task
        
    if all_latencies:
        avg_lat = sum(all_latencies) / len(all_latencies)
        p95_lat = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
        print(f"\n--- LATENCY SUMMARY ---")
        print(f"Average Turn Latency: {avg_lat*1000:.1f}ms")
        print(f"95th Percentile (p95) Latency: {p95_lat*1000:.1f}ms")
    else:
        print("\nNo transcript updates received. Check backend logs.")

if __name__ == "__main__":
    asyncio.run(main())
