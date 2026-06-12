# Concurrency, Throughput & VRAM — Nemotron ASR Pipeline

> **Scope:** Mathematical analysis of concurrent request capacity, requests-per-minute
> (RPM), throughput, and GPU memory budgets for the non-streaming Nemotron ASR
> pipeline running on **2 × NVIDIA L4 (23 GiB each)** with **16 vCPUs / 62 GiB RAM**.
> Covers the current serialised design and the theoretical maximum if concurrency
> locks are removed.

---

## 1. Hardware inventory

| Resource | Value |
|---|---|
| GPUs | 2 × NVIDIA L4 |
| VRAM per GPU | 23,034 MiB ≈ **22.5 GiB** |
| Total VRAM | 46,068 MiB ≈ **45.0 GiB** |
| CUDA context overhead | ~300–500 MiB per GPU |
| **Usable VRAM per GPU** | **~22,200 MiB** |
| CPU cores | 16 vCPUs |
| System RAM | 62 GiB |

---

## 2. Pipeline architecture

Each job in [`pipeline.py`](../pipeline.py) executes **sequentially** through
four stages:

```
 ┌───────────┐     ┌──────────┐     ┌───────────┐     ┌─────────────┐
 │ Normalize  │────▶│  Denoise  │────▶│  Diarize   │────▶│ Transcribe  │
 │  (CPU)     │     │  (CPU)    │     │  (GPU:1)   │     │  (GPU:0)    │
 │  ffmpeg    │     │  DFN3     │     │  VAD+Embed │     │ Nemotron    │
 │            │     │           │     │  +Cluster  │     │ 0.6B        │
 └───────────┘     └──────────┘     └───────────┘     └─────────────┘
```

| Stage | Component | Device | Key constraint |
|---|---|---|---|
| 1 — Normalize | ffmpeg | CPU | Subprocess; ~1 core |
| 2 — Denoise | DeepFilterNet3 | CPU | `threading.Lock` in [`audio_processing.py:67`](../audio_processing.py) |
| 3 — Diarize | ClusteringDiarizer (MarbleNet VAD + TitaNet-Large embeddings + spectral clustering) | **cuda:1** | New model instance per call; [`diarize_inventory.py:34`](../diarize_inventory.py) |
| 4 — Transcribe | Nemotron-3.5-ASR-streaming-0.6B | **cuda:0** | `threading.Lock` in [`nemotron_streaming.py:20`](../nemotron_streaming.py) |

**Current concurrency limiters:**
- `ThreadPoolExecutor(max_workers=1)` — [`pipeline.py:14`](../pipeline.py)
- `self.lock` on ASR — [`nemotron_streaming.py:12`](../nemotron_streaming.py)
- `self.lock` on DeepFilterNet — [`audio_processing.py:59`](../audio_processing.py)

---

## 3. Model VRAM budgets

### 3.1 GPU 0 — Nemotron ASR (`cuda:0`)

Model: `nvidia/nemotron-3.5-asr-streaming-0.6b` — 600 M parameters, loaded in
FP32 via [`nemotron_streaming.py:13`](../nemotron_streaming.py).

| Component | Calculation | VRAM |
|---|---|---|
| Model weights (FP32) | 600 M × 4 bytes | 2,400 MiB |
| PyTorch buffers & overhead | ~10–15 % of weights | ~300 MiB |
| **Static total** | | **~2,700 MiB** |

Dynamic memory **per inference** (one audio file at a time, chunk-wise
streaming through the encoder):

| Component | Estimate |
|---|---|
| Input chunk tensors | 10–50 MiB |
| Encoder cache (channel + time) | 100–300 MiB |
| RNNT decoder joint/prediction | 50–100 MiB |
| Intermediate activations (`inference_mode`) | 200–500 MiB |
| **Dynamic total per inference** | **~400–950 MiB (midpoint ≈ 700 MiB)** |

**Total per concurrent inference = 2,700 + 700 ≈ 3,400 MiB** (first instance).
Additional instances on the **same model object** cost only the dynamic
portion (~700 MiB each) because weights are shared.

### 3.2 GPU 1 — Diarization pipeline (`cuda:1`)

| Sub-model | Parameters | FP32 weights | With overhead |
|---|---|---|---|
| MarbleNet VAD | 91.5 K | 0.4 MiB | ~50 MiB |
| TitaNet-Large | 23 M | 92 MiB | ~150 MiB |
| **Static total** | | | **~200 MiB** |

Dynamic memory per diarization job (depends on audio length and
`batch_size=64` in [`diarize_inventory.py:23`](../diarize_inventory.py)):

| Phase | Estimate |
|---|---|
| VAD frame inference (batch=64) | 200–400 MiB |
| TitaNet multi-scale embedding extraction (batch=64) | 500–2,000 MiB |
| Spectral clustering | CPU-only |
| **Dynamic total per diarization** | **~700–2,400 MiB (midpoint ≈ 1,500 MiB)** |

**Total per concurrent diarization ≈ 200 + 1,500 = 1,700 MiB.**

> **Note:** The current code creates a new `ClusteringDiarizer` instance per
> call, so each concurrent job loads its own copy of VAD + TitaNet weights.
> If model loading were shared, only dynamic memory would add per job.

### 3.3 CPU — Normalize + Denoise

| Component | CPU | RAM |
|---|---|---|
| ffmpeg normalize | ~1 core | ~50–100 MiB per file |
| DeepFilterNet3 | ~1 core (single-threaded) | ~200–400 MiB per instance |
| **Total per job** | ~2 cores | ~300–500 MiB |

---

## 4. Maximum concurrency — VRAM-bound analysis

### Formula

```
Max_concurrent = min(
    C_gpu0,       # GPU 0 memory bound
    C_gpu1,       # GPU 1 memory bound
    C_cpu,        # CPU core bound
    C_ram         # System RAM bound
)
```

Where:

```
               ⌊ (VRAM_gpu0 − ASR_static) / ASR_dynamic ⌋   (shared weights)
C_gpu0     =  {
               ⌊ VRAM_gpu0 / (ASR_static + ASR_dynamic) ⌋   (separate instances)

               ⌊ VRAM_gpu1 / Diarizer_per_job ⌋
C_gpu1     =

C_cpu      =   ⌊ CPU_cores / cores_per_job ⌋

C_ram      =   ⌊ Available_RAM / RAM_per_job ⌋
```

### Plugging in numbers

#### Scenario A — Shared model weights (remove lock, keep single model instance)

| Bound | Calculation | Result |
|---|---|---|
| GPU 0 | ⌊(22,200 − 2,700) / 700⌋ | **27** |
| GPU 1 | ⌊22,200 / 1,700⌋ | **13** |
| CPU | ⌊16 / 2⌋ | **8** |
| RAM | ⌊49,000 / 400⌋ | **122** (not binding) |
| **Effective max** | min(27, 13, 8, 122) | **8** (CPU-bound) |

#### Scenario B — Separate model instances per worker

| Bound | Calculation | Result |
|---|---|---|
| GPU 0 | ⌊22,200 / 3,400⌋ | **6** |
| GPU 1 | ⌊22,200 / 1,700⌋ | **13** |
| CPU | ⌊16 / 2⌋ | **8** |
| RAM | ⌊49,000 / 400⌋ | **122** |
| **Effective max** | min(6, 13, 8, 122) | **6** (GPU 0-bound) |

### Summary table

| Scenario | Bottleneck | Max concurrent |
|---|---|---|
| A — Shared weights, lock removed | CPU cores (16 / 2) | **8** |
| B — Separate model instances | GPU 0 VRAM | **6** |
| Current code (all locks in place) | `max_workers=1` | **1** |

---

## 5. RPM and throughput relationship

### Core definitions

```
RPM  = Requests Per Minute (jobs successfully completed per minute)
T    = average processing time per job (seconds)
C    = effective concurrency (number of simultaneous jobs)
```

### Fundamental formula

```
RPM = (C × 60) / T
```

Or equivalently:

```
Throughput (jobs/sec) = C / T
RPM = Throughput × 60
```

### Per-stage latency estimates (for a typical 60-second audio file)

| Stage | Typical latency | Notes |
|---|---|---|
| Normalize (ffmpeg) | 1–3 s | CPU, single-threaded subprocess |
| Denoise (DFN3) | 2–5 s | CPU, single-threaded; scales with audio length |
| Diarize | 8–15 s | GPU:1; VAD + embedding + clustering |
| Transcribe (Nemotron) | 10–20 s | GPU:0; chunk-wise streaming through whole file |
| **Total per job (T)** | **~25–40 s (midpoint ≈ 30 s)** | |

### RPM projections

| Concurrency (C) | T = 30 s | T = 40 s | T = 20 s |
|---|---|---|---|
| **1** (current) | 2.0 RPM | 1.5 RPM | 3.0 RPM |
| **4** (conservative) | 8.0 RPM | 6.0 RPM | 12.0 RPM |
| **6** (Scenario B max) | 12.0 RPM | 9.0 RPM | 18.0 RPM |
| **8** (Scenario A max) | 16.0 RPM | 12.0 RPM | 24.0 RPM |

### RPM ↔ VRAM relationship

The relationship is **step-wise**, not linear. Adding concurrency consumes VRAM
in fixed increments:

```
VRAM_gpu0(C) = ASR_static + C × ASR_dynamic
             = 2,700 + C × 700     (MiB, shared weights)

VRAM_gpu1(C) = C × Diarizer_per_job
             = C × 1,700            (MiB, separate instances)

RPM(C)       = (C × 60) / T
```

Combining:

```
                  (VRAM_gpu0 − ASR_static)
RPM_max_gpu0  =  ─────────────────────────  ×  60 / T
                       ASR_dynamic

                  (22,200 − 2,700)
              =  ─────────────────  ×  60 / 30
                       700

              =  27.9  ×  2  =  55.7 RPM    (theoretical GPU 0 ceiling)


                   VRAM_gpu1
RPM_max_gpu1  =  ──────────────  ×  60 / T
                  Diar_per_job

              =  22,200 / 1,700  ×  60 / 30

              =  13.1  ×  2  =  26.1 RPM    (theoretical GPU 1 ceiling)


RPM_max_cpu   =  (16 / 2)  ×  60 / 30  =  16 RPM    (CPU ceiling)
```

**The binding constraint is CPU: RPM ≤ 16** with shared weights on this VM.

### VRAM utilisation at each concurrency level

| C | GPU 0 VRAM | GPU 0 % | GPU 1 VRAM | GPU 1 % | RPM (T=30s) |
|---|---|---|---|---|---|
| 1 | 3,400 MiB | 15% | 1,700 MiB | 8% | 2.0 |
| 2 | 4,100 MiB | 18% | 3,400 MiB | 15% | 4.0 |
| 4 | 5,500 MiB | 25% | 6,800 MiB | 31% | 8.0 |
| 6 | 6,900 MiB | 31% | 10,200 MiB | 46% | 12.0 |
| 8 | 8,300 MiB | 37% | 13,600 MiB | 61% | 16.0 |
| 10 | 9,700 MiB | 44% | 17,000 MiB | 77% | 20.0 |
| 13 | 11,800 MiB | 53% | 22,100 MiB | 100% | 26.0 |

> **Warning:** At C ≥ 10, GPU 1 utilisation exceeds 75% and memory
> fragmentation becomes a real risk. Peak spikes from long audio files can
> push actual usage above the midpoint estimates.

---

## 6. What changes if locks and `max_workers=1` are removed

### 6.1 Code changes required

#### `pipeline.py` — increase worker pool

```diff
 # pipeline.py line 14
-self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemotron-job")
+self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="nemotron-job")
```

#### `nemotron_streaming.py` — remove or relax the lock

```diff
 # nemotron_streaming.py line 20-31
-        with self.lock:
-            self.torch.cuda.set_device(self.device)
-            ...
+        # Option A: remove lock entirely (requires inference_mode safety)
+        self.torch.cuda.set_device(self.device)
+        ...
```

> **Caution on Option A:** The Nemotron conformer model uses mutable encoder
> cache state (`cache_last_channel`, `cache_last_time`, etc.) that is created
> fresh per call, so concurrent `transcribe()` calls on the same model instance
> are safe **as long as each call uses its own local cache variables** — which
> the current code already does (variables are local to `transcribe()`). The
> `self.lock` is conservative; removing it is safe for this specific call pattern.

> **Caution:** `self.model.encoder.set_default_att_context_size()` and
> `self.model.set_inference_prompt()` mutate shared model state. If all
> concurrent calls use the **same** `chunk_ms` and `language`, this is fine.
> If they differ, you need per-call isolation (e.g., model copies or a
> semaphore that serialises only the setup + forward pass).

#### `audio_processing.py` — DeepFilterNet lock

The `self.lock` on DeepFilterNet in [`audio_processing.py:67`](../audio_processing.py)
**cannot be safely removed** because `enhance()` mutates `self.state` (the
DFN3 state buffer). Options:

- **Keep the lock** — denoise becomes the CPU-side serialisation point.
- **Pool multiple `DeepFilterNetDenoiser` instances** — each with its own
  `model` + `state`, assigned round-robin to workers.

### 6.2 Before vs after comparison

| Metric | Before (current) | After (C=8, shared ASR) |
|---|---|---|
| `max_workers` | 1 | 8 |
| ASR lock | `threading.Lock` (serial) | Removed (concurrent) |
| Concurrent jobs | 1 | Up to 8 |
| GPU 0 peak VRAM | ~3,400 MiB (15%) | ~8,300 MiB (37%) |
| GPU 1 peak VRAM | ~1,700 MiB (8%) | ~13,600 MiB (61%) |
| RPM (T=30s) | **2.0** | **16.0** |
| RPM improvement | — | **8×** |
| Risk | None (serial) | Memory fragmentation; long-audio OOM on GPU 1 |

### 6.3 Pipeline parallelism (bonus)

The current design processes stages sequentially within each job. But across
jobs, **different stages can overlap** when concurrency > 1:

```
Time ──────────────────────────────────────────────────────▶

Job 1:  [Normalize][Denoise][──Diarize──][───Transcribe───]
Job 2:           [Normalize][Denoise][──Diarize──][───Transcribe───]
Job 3:                    [Normalize][Denoise][──Diarize──][───Transcr...]

                                     ▲            ▲
                                  GPU:1 busy    GPU:0 busy
                                  in parallel   in parallel
```

This naturally happens with `ThreadPoolExecutor(max_workers=N)` because each
worker runs the full pipeline, and GPU operations from different workers
overlap via CUDA's internal scheduling.

---

## 7. Recommended production configuration

### Conservative (safe for all audio lengths)

```python
# pipeline.py
self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nemotron-job")
```

| Metric | Value |
|---|---|
| Concurrent jobs | 4 |
| GPU 0 peak | ~5,500 MiB (25%) |
| GPU 1 peak | ~6,800 MiB (31%) |
| RPM (T=30s) | 8.0 |
| Headroom | >60% on both GPUs |

### Aggressive (for short audio ≤ 2 min)

```python
# pipeline.py
self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="nemotron-job")
```

| Metric | Value |
|---|---|
| Concurrent jobs | 8 |
| GPU 0 peak | ~8,300 MiB (37%) |
| GPU 1 peak | ~13,600 MiB (61%) |
| RPM (T=30s) | 16.0 |
| Headroom | ~39% on GPU 1 |

### FP16 upgrade path (future)

If the ASR model is loaded in **FP16** instead of FP32:

```
ASR_static  = 600M × 2 bytes = 1,200 MiB  (vs 2,400 MiB)
ASR_dynamic ≈ 350 MiB                      (vs 700 MiB)
C_gpu0      = ⌊(22,200 − 1,500) / 350⌋ = 59  (vs 27)
```

FP16 nearly doubles the GPU 0 concurrency ceiling. The bottleneck shifts
entirely to CPU and GPU 1.

---

## 8. Quick-reference formulae

```
┌──────────────────────────────────────────────────────────────────┐
│  RPM = (C × 60) / T                                             │
│                                                                  │
│  C = min(C_gpu0, C_gpu1, C_cpu, C_ram)                          │
│                                                                  │
│  C_gpu0 = ⌊(VRAM_0 − W_asr) / D_asr⌋     (shared weights)     │
│         = ⌊VRAM_0 / (W_asr + D_asr)⌋      (separate instances) │
│                                                                  │
│  C_gpu1 = ⌊VRAM_1 / (W_diar + D_diar)⌋                         │
│                                                                  │
│  C_cpu  = ⌊cores / cores_per_job⌋                               │
│                                                                  │
│  Throughput = C / T  (jobs/sec)                                  │
│  RPM = Throughput × 60                                           │
│                                                                  │
│  VRAM_gpu0(C) = W_asr + C × D_asr                               │
│  VRAM_gpu1(C) = C × (W_diar + D_diar)                           │
│                                                                  │
│  Where:                                                          │
│    W = static weight memory     D = dynamic per-inference memory │
│    T = avg job latency (sec)    C = concurrency level            │
│    VRAM_0 = 22,200 MiB          VRAM_1 = 22,200 MiB             │
│    W_asr = 2,700 MiB            D_asr = 700 MiB                 │
│    W_diar = 200 MiB             D_diar = 1,500 MiB              │
│    cores = 16                   cores_per_job = 2                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. Files referenced

| File | Role |
|---|---|
| [`pipeline.py`](../pipeline.py) | Job orchestration; `ThreadPoolExecutor(max_workers=1)` |
| [`nemotron_streaming.py`](../nemotron_streaming.py) | ASR model loading + inference; `threading.Lock` |
| [`audio_processing.py`](../audio_processing.py) | DeepFilterNet denoiser; `threading.Lock` |
| [`diarize_inventory.py`](../diarize_inventory.py) | ClusteringDiarizer (VAD + TitaNet + clustering) |
| [`app.py`](../app.py) | FastAPI endpoints; calls `JobStore.submit()` |
