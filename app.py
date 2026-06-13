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

import csv, io, re, shutil, sys, tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from biasing_context import is_hindi
from pipeline import JobStore

app=FastAPI(title="Nemotron Streaming ASR Test")
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
def _biasing_context(name,institute_name,total_due,due_date):
    """Bundle optional per-call metadata for Hindi context biasing (None if all blank)."""
    ctx={k:v for k,v in {"name":name,"institute_name":institute_name,"total_due":total_due,"due_date":due_date}.items() if v}
    return ctx or None
@app.post("/api/jobs")
def create_job(file:UploadFile=File(...),language:str=Form("hi-IN"),chunk_ms:int=Form(1120),itn_backend:str=Form("custom"),name:str=Form(None),institute_name:str=Form(None),total_due:str=Form(None),due_date:str=Form(None)):
    if chunk_ms not in {80,160,320,560,1120}:raise HTTPException(400,"Unsupported chunk size")
    if itn_backend not in {"custom","nemo","compare"}:raise HTTPException(400,"Unsupported ITN backend")
    with tempfile.NamedTemporaryFile(delete=False,suffix=Path(file.filename or "audio.webm").suffix) as temp:
        shutil.copyfileobj(file.file,temp); path=Path(temp.name)
    try:return store.submit(path,file.filename or "recording.webm",language,chunk_ms,itn_backend,_biasing_context(name,institute_name,total_due,due_date))
    finally:path.unlink(missing_ok=True)
@app.post("/api/jobs/sample/{filename}")
def create_sample_job(filename:str,language:str=Form("hi-IN"),chunk_ms:int=Form(1120),itn_backend:str=Form("custom"),name:str=Form(None),institute_name:str=Form(None),total_due:str=Form(None),due_date:str=Form(None)):
    source=(ROOT/"recording"/Path(filename).name)
    if not source.exists():raise HTTPException(404,"Sample not found")
    if itn_backend not in {"custom","nemo","compare"}:raise HTTPException(400,"Unsupported ITN backend")
    return store.submit(source,source.name,language,chunk_ms,itn_backend,_biasing_context(name,institute_name,total_due,due_date))
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

@app.websocket("/api/stream")
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
        flush_signal=flush_signal
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
    
    # Force vad to be false for this endpoint as requested
    vad = "false"
    
    # Map flush_signal query parameter (defaults to "false")
    flush_signal = websocket.query_params.get("flush_signal") or "false"
    
    # Run in Immediate Mode (chunk_ms=0, denoise=true)
    await handle_websocket_stream(
        websocket=websocket,
        store=store,
        language=language,
        denoise="true",
        vad=vad,
        diarize="false",
        input_rate=input_rate,
        chunk_ms=0,
        call_code=call_code,
        flush_signal=flush_signal
    )


