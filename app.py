from __future__ import annotations
import os
import logging
import warnings
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "nemo", "NeMo"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import csv, io, re, shutil, sys, tempfile, asyncio
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from utils.pipeline import JobStore, resolve_model
from streaming_handler import log_event
import uuid
import time
from datetime import datetime
from starlette.concurrency import iterate_in_threadpool

def is_hindi(language: str) -> bool:
    return (language or "").strip().lower().startswith("hi")

def deprecated(func):
    import functools
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"[WARNING] Client connected to deprecated endpoint '{func.__name__}'. Migrate to /ws/stt.", flush=True)
        return await func(*args, **kwargs)
    return wrapper

app=FastAPI(title="Nemotron Streaming ASR Test")

# Allow large audio file uploads (up to 500 MB) — telephony call recordings
# can exceed Starlette's default 1MB multipart limit
try:
    from starlette.middleware.base import BaseHTTPMiddleware  # noqa
    app.state.max_upload_size = 500 * 1024 * 1024  # 500 MB
except Exception:
    pass

@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # Check content type of request
    content_type = request.headers.get("content-type", "").lower()
    body_str = ""
    if "multipart/" in content_type:
        body_str = "[Multipart Form Data]"
    else:
        try:
            body_bytes = await request.body()
            # Restore request receive stream so downstream handler can consume it
            async def receive():
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            request._receive = receive
            
            if body_bytes:
                body_str = body_bytes.decode("utf-8", errors="replace")
                if len(body_str) > 1000:
                    body_str = body_str[:1000] + "... [truncated]"
            else:
                body_str = "[empty]"
        except Exception as e:
            body_str = f"[Error reading body: {e}]"
            
    client_host = request.client.host if request.client else "unknown"
    log_event(f"[{now_str}] [API] [REQ] [ID: {request_id}] {request.method} {request.url.path} | Client: {client_host} | Query: {request.url.query} | Body: {body_str}")
    
    t0 = time.time()
    try:
        response = await call_next(request)
        duration = (time.time() - t0) * 1000.0
        
        # Determine if we should read response body
        is_json = "application/json" in response.headers.get("content-type", "")
        content_length = response.headers.get("content-length")
        is_small = True
        if content_length and int(content_length) > 50000:
            is_small = False
            
        body_to_log = ""
        if is_json and is_small:
            try:
                response_body = [chunk async for chunk in response.body_iterator]
                response.body_iterator = iterate_in_threadpool(iter(response_body))
                joined_body = b"".join(response_body)
                body_to_log = joined_body.decode("utf-8", errors="replace")
                if len(body_to_log) > 1000:
                    body_to_log = body_to_log[:1000] + "... [truncated]"
            except Exception as e:
                body_to_log = f"[Error reading response: {e}]"
        else:
            body_to_log = f"[Content-Type: {response.headers.get('content-type', 'unknown')}]"
            
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_event(f"[{now_str}] [API] [RES] [ID: {request_id}] Status: {response.status_code} | Duration: {duration:.2f}ms | Response: {body_to_log}")
        return response
    except Exception as e:
        duration = (time.time() - t0) * 1000.0
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_event(f"[{now_str}] [API] [RES] [ID: {request_id}] Exception: {e} | Duration: {duration:.2f}ms")
        raise e
store=JobStore(); ROOT=Path(__file__).parent
csv.field_size_limit(sys.maxsize)  # transcript columns hold raw beam-search dumps (100k+ chars)

@app.on_event("startup")
def startup_event():
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(nemo_logging.logging.ERROR)
    except Exception:
        pass
    print("Pre-loading models on startup...")
    store._ensure_models()
    print("Models pre-loaded successfully!")

_TURN_SPLIT=re.compile(r"(customer|agent):\s*(?=\[Hypothesis)")
_HYP_TEXT=re.compile(r"text='(.*?)', dec_out=", re.DOTALL)
_PLAIN_TURN=re.compile(r"^(customer|agent):[ \t]*(.*)$", re.MULTILINE)
def parse_dialogue(raw:str)->list[dict]:
    """Turn a stored transcript into readable chat turns, in either format:

    - clean `speaker: text` lines (current eval output, one turn per line), or
    - the legacy `speaker: [Hypothesis(...), ...]` beam dump, where each speaker turn holds
      one or more beam lists (one per streamed chunk) and we keep the top non-empty
      hypothesis of each chunk, joined into the spoken utterance.
    """
    raw=raw or ""
    if "[Hypothesis" not in raw:  # clean transcript — just split the speaker-labelled lines
        return [{"speaker":m.group(1),"text":m.group(2).strip()}
                for m in _PLAIN_TURN.finditer(raw) if m.group(2).strip()]
    turns=[]
    parts=_TURN_SPLIT.split(raw)
    for i in range(1,len(parts)-1,2):
        speaker=parts[i]
        chunks=(next((t for t in _HYP_TEXT.findall(beam) if t.strip()),"") for beam in re.split(r"\]\s*\[",parts[i+1]))
        text=" ".join(c for c in chunks if c).strip()
        if text:turns.append({"speaker":speaker,"text":text})
    return turns
app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static")
@app.get("/")
def index():return FileResponse(ROOT/"static"/"index.html")
@app.get("/api/samples")
def samples():return [{"name":p.name,"size":p.stat().st_size} for p in sorted((ROOT/"recording").glob("*")) if p.is_file()]

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
def healthz():
    return Response("ok", media_type="text/plain")
@app.get("/api/evaluations/{filename}")
def evaluation(filename:str):
    path=ROOT/Path(filename).name
    if not path.name.endswith("_hindi_biasing_normalized.csv") or not path.exists():
        raise HTTPException(404,"Evaluation not found")
    with path.open(encoding="utf-8",newline="") as fh:
        rows=list(csv.DictReader(fh))
    # Per-row call recordings: newer eval runs embed `cr_recording_url` in each row (use it
    # directly). Older normalized CSVs dropped it, so we fall back to the source CSV — but the
    # eval keeps only the HINDI rows and renumbers them 1..N, so we MUST apply the same language
    # filter before indexing by idx. Skipping it lets non-Hindi source rows shift every URL, so
    # the player ends up on a different debtor's call than the transcript shown beside it.
    source=ROOT/path.name.replace("_hindi_biasing_normalized.csv",".csv")
    urls=[]
    if source.exists():
        src_rows=list(csv.DictReader(source.open(encoding="utf-8",newline="")))
        if any(r.get("language") for r in src_rows):  # language column present -> filter like the eval did
            src_rows=[r for r in src_rows if is_hindi(r.get("language"))]
        urls=[r.get("cr_recording_url","") for r in src_rows]
    def truthy(value):return str(value).strip().lower()=="true"
    def recording_url(row):
        embedded=(row.get("cr_recording_url") or "").strip()
        if embedded:return embedded
        idx=row.get("idx","")
        i=int(idx)-1 if str(idx).isdigit() else -1
        return urls[i] if 0<=i<len(urls) else ""
    compact=[]
    for row in rows:
        compact.append({
            **{key:row.get(key,"") for key in ("idx","name","institute","brand","total_due","due_date","dynamic_phrases")},
            **{key:truthy(row.get(key)) for key in ("biasing_applied","name_hit_baseline","name_hit_biased","brand_hit_baseline","brand_hit_biased")},
            "recording_url":recording_url(row),
            "baseline_turns":parse_dialogue(row.get("baseline_transcript","")),
            "biased_turns":parse_dialogue(row.get("biased_transcript","")),
        })
    return {
        "filename":path.name,
        "rows":compact,
        "summary":{
            "total":len(compact),
            "biasing_applied":sum(r["biasing_applied"] for r in compact),
            "name_hit_baseline":sum(r["name_hit_baseline"] for r in compact),
            "name_hit_biased":sum(r["name_hit_biased"] for r in compact),
            "brand_hit_baseline":sum(r["brand_hit_baseline"] for r in compact),
            "brand_hit_biased":sum(r["brand_hit_biased"] for r in compact),
        },
    }
@app.post("/api/jobs")
async def create_job(file:UploadFile=File(...),language:str=Form("hi-IN"),chunk_ms:int=Form(1120),itn_backend:str=Form("custom"),model:str=Form("sarga:v1"),diarize:bool=Form(True),denoise:bool=Form(True)):
    # Allow any chunk_ms — clamp to nearest valid model window if needed
    valid_chunks = {80, 160, 320, 560, 1120}
    if chunk_ms not in valid_chunks:
        chunk_ms = min(valid_chunks, key=lambda v: abs(v - chunk_ms))
    if itn_backend not in {"custom","nemo","compare","none"}:raise HTTPException(400,"Unsupported ITN backend")
    model = resolve_model(model)
    if model not in {"sarga:v1", "credresolve:v1"}:raise HTTPException(400,"Unsupported model name")
    path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(file.filename or "audio.webm").suffix) as temp:
            temp.write(content)
            path = Path(temp.name)
        return store.submit(path,file.filename or "recording.webm",language,chunk_ms,itn_backend,model,diarize,denoise)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {type(e).__name__}: {e}")
    finally:
        if path:
            path.unlink(missing_ok=True)
@app.post("/api/jobs/sample/{filename}")
def create_sample_job(filename:str,language:str=Form("hi-IN"),chunk_ms:int=Form(1120),itn_backend:str=Form("custom"),model:str=Form("sarga:v1"),diarize:bool=Form(True),denoise:bool=Form(True)):
    source=(ROOT/"recording"/Path(filename).name)
    if not source.exists():raise HTTPException(404,"Sample not found")
    if itn_backend not in {"custom","nemo","compare","none"}:raise HTTPException(400,"Unsupported ITN backend")
    model = resolve_model(model)
    if model not in {"sarga:v1", "credresolve:v1"}:raise HTTPException(400,"Unsupported model name")
    return store.submit(source,source.name,language,chunk_ms,itn_backend,model,diarize,denoise)
@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    job=store.public(job_id)
    if not job:raise HTTPException(404,"Job not found")
    return job
@app.get("/api/jobs/{job_id}/audio/{kind}")
def get_audio(job_id:str,kind:str):
    path=store.audio_path(job_id,kind)
    if not path:raise HTTPException(404,"Audio not found")
    return FileResponse(path,media_type="audio/wav")

from fastapi import WebSocket
from streaming_handler import handle_websocket_stream

# DEPRECATED: Use /ws/stt instead
@app.websocket("/api/stream")
@deprecated
async def websocket_stream(
    websocket: WebSocket,
    language: str = "hi-IN",
    denoise: str = "true",
    vad: str = "true",
    diarize: str = "false",
    input_rate: int = 16000,
    chunk_ms: int = 320,
    call_code: str = None,
    flush_signal: str = "false"
):
    if not call_code:
        call_code = websocket.query_params.get("call_code") or websocket.query_params.get("call-code")
    if not flush_signal or flush_signal == "false":
        flush_signal = websocket.query_params.get("flush_signal") or "false"
    model = websocket.query_params.get("model") or "sarga:v1"
    itn = websocket.query_params.get("itn") or "false"
    itn_backend = websocket.query_params.get("itn_backend") or "custom"
    await handle_websocket_stream(
        websocket=websocket,
        store=store,
        language=language,
        denoise=denoise,
        vad=vad,
        diarize=diarize,
        input_rate=input_rate,
        chunk_ms=chunk_ms,
        call_code=call_code,
        flush_signal=flush_signal,
        model=model,
        itn=itn,
        itn_backend=itn_backend
    )

@app.websocket("/ws/stt")
async def ws_stt(
    websocket: WebSocket,
):
    # Extract query parameters to match production gateway URL style
    lang_code = websocket.query_params.get("language-code") or "hi-IN"
    if lang_code == "hi":
        language = "hi-IN"
    elif lang_code == "en":
        language = "en-US"
    else:
        language = lang_code
        
    sample_rate_str = websocket.query_params.get("sample_rate") or "16000"
    input_rate = int(sample_rate_str) if sample_rate_str.isdigit() else 16000
    
    call_code = websocket.query_params.get("call_code") or websocket.query_params.get("call-code")
    
    # Read VAD from query param — if client passes flush_signal=true (telephony mode)
    # enable server-side Silero VAD to auto-detect speech boundaries
    flush_signal = websocket.query_params.get("flush_signal") or "false"
    # Enable VAD when flush_signal=true so server auto-flushes on speech end
    vad = "true" if flush_signal.lower() == "true" else (websocket.query_params.get("vad") or "false")
    
    model = websocket.query_params.get("model") or "sarga:v1"
    diarize = websocket.query_params.get("diarize") or "false"
    itn = websocket.query_params.get("itn") or "false"
    itn_backend = websocket.query_params.get("itn_backend") or "custom"
    # Read denoise from query param (default true for quality)
    denoise = websocket.query_params.get("denoise") or "true"
    
    # Use chunk_ms=80 for optimal internal inference windowing (matches batch API)
    await handle_websocket_stream(
        websocket=websocket,
        store=store,
        language=language,
        denoise=denoise,
        vad=vad,
        diarize=diarize,
        input_rate=input_rate,
        chunk_ms=80,
        call_code=call_code,
        flush_signal=flush_signal,
        model=model,
        itn=itn,
        itn_backend=itn_backend
    )


def _make_public_audio_link(request: Request, job_id: str) -> str:
    base = str(request.base_url).rstrip('/')
    if "localhost" in base or "127.0.0.1" in base or ":8000" in base:
        base = "http://13.234.132.26:8002"
    return f"{base}/api/jobs/{job_id}/audio/normalized"


from pydantic import BaseModel

class CSVRequest(BaseModel):
    job_ids: list[str]


@app.get("/api/jobs/batch/csv")
def get_batch_csv(job_ids: str, request: Request):
    ids = [jid.strip() for jid in job_ids.split(",") if jid.strip()]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["filename", "audio_link", "language_id", "transcript"])
    for jid in ids:
        job = store.public(jid)
        if job:
            filename = job.get("filename", "audio.wav")
            status = job.get("status")
            language_id = "unknown"
            if status == "complete":
                result = job.get("result") or {}
                transcript = result.get("transcript") or ""
                language_id = result.get("language_id") or job.get("language") or "hi-IN"
            elif status == "failed":
                transcript = f"ERROR: {job.get('error')}"
                language_id = job.get("language") or "hi-IN"
            else:
                transcript = f"STATUS: {status}"
                language_id = job.get("language") or "hi-IN"
            audio_link = _make_public_audio_link(request, jid)
            writer.writerow([filename, audio_link, language_id, transcript])
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_transcript.csv"}
    )


@app.post("/api/jobs/batch/csv")
def get_batch_csv_post(request: Request, body: CSVRequest):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["filename", "audio_link", "language_id", "transcript"])
    for jid in body.job_ids:
        job = store.public(jid)
        if job:
            filename = job.get("filename", "audio.wav")
            status = job.get("status")
            language_id = "unknown"
            if status == "complete":
                result = job.get("result") or {}
                transcript = result.get("transcript") or ""
                language_id = result.get("language_id") or job.get("language") or "hi-IN"
            elif status == "failed":
                transcript = f"ERROR: {job.get('error')}"
                language_id = job.get("language") or "hi-IN"
            else:
                transcript = f"STATUS: {status}"
                language_id = job.get("language") or "hi-IN"
            audio_link = _make_public_audio_link(request, jid)
            writer.writerow([filename, audio_link, language_id, transcript])
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_transcript.csv"}
    )


@app.post("/api/jobs/batch")
async def create_batch_job(
    request: Request,
    files: list[UploadFile] = File(...),
    language: str = Form("hi-IN"),
    chunk_ms: int = Form(1120),
    itn_backend: str = Form("custom"),
    model: str = Form("sarga:v1"),
    diarize: bool = Form(True),
    denoise: bool = Form(True)
):
    if chunk_ms not in {80, 160, 320, 560, 1120}:
        raise HTTPException(400, "Unsupported chunk size")
    if itn_backend not in {"custom", "nemo", "compare", "none"}:
        raise HTTPException(400, "Unsupported ITN backend")
    model = resolve_model(model)
    if model not in {"sarga:v1", "credresolve:v1"}:
        raise HTTPException(400, "Unsupported model name")
 
    job_ids = []
    for file in files:
        suffix = Path(file.filename or "audio.webm").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            path = Path(temp.name)
        try:
            job = store.submit(
                path,
                file.filename or "recording.webm",
                language,
                chunk_ms,
                itn_backend,
                model,
                diarize,
                denoise
            )
            job_ids.append((job["id"], file.filename or "recording.webm"))
        finally:
            path.unlink(missing_ok=True)

    completed = False
    while not completed:
        completed = True
        for job_id, _ in job_ids:
            job = store.public(job_id)
            if not job or job.get("status") not in {"complete", "failed"}:
                completed = False
                break
        if not completed:
            await asyncio.sleep(0.5)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["filename", "audio_link", "language_id", "transcript"])
    for job_id, filename in job_ids:
        job = store.public(job_id)
        language_id = "unknown"
        if job and job.get("status") == "complete":
            result = job.get("result") or {}
            transcript = result.get("transcript") or ""
            language_id = result.get("language_id") or job.get("language") or "hi-IN"
        elif job and job.get("status") == "failed":
            transcript = f"ERROR: {job.get('error')}"
            language_id = job.get("language") or "hi-IN"
        else:
            transcript = "ERROR: Job data lost"
            language_id = language
        audio_link = _make_public_audio_link(request, job_id)
        writer.writerow([filename, audio_link, language_id, transcript])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_transcript.csv"}
    )



