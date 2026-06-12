# Dynamic Per-Row Context Biasing — Integration Guide

## Overview

This document covers the per-row context biasing feature added for testing
`nvidia/nemotron-3.5-asr-streaming-0.6b` on Hindi debt-collection recordings.

The goal: improve transcription of four call-specific fields — **debtor name**,
**lender brand**, **amount owed**, and **due date** — by telling the decoder to
prefer those phrases *before* each recording is transcribed. Since every row in
the test CSVs belongs to a different debtor, the phrase list is rebuilt from
scratch for every row at runtime. No code change is needed when new rows are
added to the CSV.

---

## Background: Why Two Tools for Four Columns

The four columns fail in different ways and need different fixes:

| Column | Failure mode | Fix |
|---|---|---|
| `name` | model **mishears** the name (wrong characters emitted) | context biasing — boosts the correct pronunciation path in the decoder |
| `institute_name` | brand mis-recognition + raw routing code (e.g. `HDBFS_PL`) | context biasing after code→brand mapping |
| `total_due` | number **formatting** (`अठारह हज़ार` → `₹18,140`) | ITN (already wired; no change needed) |
| `due_date` | date formatting | ITN (already wired; no change needed) |

Context biasing works at the **acoustic decoding** stage — it boosts log-probabilities
for specific token sequences so the model is more likely to emit the target phrase
even when the audio is noisy. ITN is a **post-processing** step that reformats
already-recognized words; it has no access to the audio and cannot recover a
mishear.

Amounts and dates go through biasing *and* ITN: biasing nudges the recognition,
ITN owns the final canonical format (e.g. `₹2,41,509`).

---

## Architecture

```
CSV row  {name, institute_name, total_due, due_date}
    │
    ▼
biasing_context.py
    ├── institute_to_brand()         HDBFS_PL → "HDB Financial"
    ├── row_to_biasing_context()     column → field-name mapping
    └── build_key_phrases_file()
            ├── parse_biasing_context()               context_assembler.py
            ├── build_request_scoped_phrase_pack()    context_assembler.py
            │       ├── generate_general_variants()   variant_generator.py
            │       ├── generate_amount_variants()    variant_generator.py
            │       ├── generate_date_variants()      variant_generator.py
            │       ├── generate_hindi_script_variants() hindi_transliteration.py
            │       ├── rank_phrase_candidates()      phrase_ranker.py
            │       └── merge with phrases/hi.txt (static Hindi lexicon)
            └── write key_phrases.txt (one phrase per line)
    │
    ▼
pipeline.py  _configure_biasing()
    └── asr.configure_biasing(key_phrases.txt)
            └── NeMo malsd_batch + beam.boosting_tree
    │
    ▼  (all speaker turns)
NemotronStreamingASR.transcribe()   [biasing active]
    │
    ▼
pipeline.py  _normalize_turn()      [ITN: amounts + dates → canonical]
    │
    ▼
result  {transcript, canonical_text, biasing metrics}
    │
finally:
    asr.reset_decoding()            [restore maes for next job]
```

---

## Key Design Decisions

### NeMo boosting tree, not CTC word-synchronous

The model decodes with RNN-T `maes`. The existing `context_biasing/` module
uses NeMo's CTC word-synchronous (`apply_context_biasing` / `context_file`)
knobs — those are inert for a transducer decode.

NeMo 2.8 ships a separate GPU boosting-tree path that *does* support
transducers, but only on **batched** strategies. The relevant constraint from
`rnnt_decoding.py`:

```
if strategy is TransducerDecodingStrategyType.MAES:
    raise NotImplementedError(
        "Model ... with strategy `maes` does not support boosting tree."
    )
```

So biased jobs switch to `malsd_batch` (the recommended batched RNN-T strategy
with LM/biasing support) for the duration of the job, then restore `maes`
afterward.

### Phrase assembly reused, CTC-WS runtime bypassed

`context_biasing/context_assembler.py`, `variant_generator.py`,
`hindi_transliteration.py`, and `phrase_ranker.py` are reused unchanged. They
import only from each other — **not** from `context_biasing/context_biasing.py`
or the missing `context_biasing/nemo_export.py` — so the phrase assembly half
works without the heavy CTC-WS runtime or a second model instance.

### Per-job, not per-turn

Biasing is applied once per CSV row (one debtor) before the diarized turn loop
and cleared in a `finally` block after the last turn. All speaker turns within
that job share the same phrase list. A phrase list takes ~50–200 ms to compile
(mostly variant generation + Hindi transliteration); applying it to the decode
config is instantaneous.

### Graceful degradation

`_configure_biasing` catches all exceptions and returns a metrics dict with
`applied: False` and the reason string. The job continues unbiased — ITN still
runs on amounts and dates — rather than failing. The probe script
(`probe_biasing.py`) confirms whether the installed NeMo build and checkpoint
support the boosting tree before committing to the integration.

---

## New and Changed Files

### `biasing_context.py` (new)

Builds a per-row key-phrases file from CSV metadata.

**`INSTITUTE_BRANDS`** — dict mapping lender routing codes to spoken brand names.
Routing suffixes (`_SPL`, `_SOUTH`, `_NACL`, `_PL`, `_RURAL`, `_AUTO`) are
stripped before lookup. Unmapped codes fall back to `title_case(code)` so
a brand-new lender still biases without a code change.

```
UGRO_CAPITAL  → "Ugro Capital"
HDBFS_PL      → "HDB Financial"   (strip _PL, map HDBFS)
FOOBAR_SPL    → "Foobar"          (unmapped fallback)
```

**`row_to_biasing_context(row)`** — maps CSV columns to `context_assembler` field names:

| CSV column | `BiasingContext` field | Notes |
|---|---|---|
| `name` | `debtor_name` | weight 120 in ranker |
| `institute_name` | `lender` | after `institute_to_brand()`; weight 105 |
| `total_due` | `amounts` | appends `" rupees"` so variant_generator emits Devanagari + word forms; skips `0` |
| `due_date` | `dates` | strips the timestamp portion `"2026-01-05 00:00:00"` → `"2026-01-05"` |

**`build_key_phrases_file(row, *, language, ...)`** — full pipeline:
1. Gates on `is_hindi(language)` — returns `None` for non-Hindi rows.
2. Builds a `BiasingContext` via `parse_biasing_context`.
3. Calls `build_request_scoped_phrase_pack` (merges dynamic candidates with
   the static `phrases/hi.txt` lexicon, ranks top-32).
4. Flattens the underscore-delimited variant groups into one phrase per line
   (NeMo's boosting tree expects plain one-phrase-per-line format).
5. Writes `key_phrases.txt` to `job_dir/biasing/`.

Returns `(Path, AssembledPhrasePack)` or `None`.

---

### `nemotron_streaming.py` (modified)

Added three constants and two methods to `NemotronStreamingASR`.

**New constants:**
```python
BIASING_STRATEGY   = "malsd_batch"  # batched RNN-T beam, supports boosting tree
BIASING_ALPHA      = 1.0            # fusion weight: boosting tree vs acoustic score
BIASING_CONTEXT_SCORE = 3.0        # per-arc log-prob boost inside the context graph
```

**`__init__` additions:**
- Saves `self._base_decoding_cfg` (the default `maes` config) immediately after
  the initial `change_decoding_strategy` call, so `reset_decoding` has a clean
  snapshot to restore.
- Initialises `self._biasing_active = False`.

**`configure_biasing(key_phrases_file, *, source_lang, alpha, context_score, use_triton)`:**

Switches the loaded model to `malsd_batch` + `beam.boosting_tree` for all
subsequent `transcribe()` calls. Must be called under `self.lock` (it acquires
the lock itself). Raises on failure — callers should fall back to unbiased decode.

The boosting tree config:
```python
BoostingTreeModelConfig(
    key_phrases_file = str(key_phrases_file),  # one phrase per line
    source_lang      = "hi",
    context_score    = 3.0,   # per-arc boost
    depth_scaling    = 2.0,   # default; correct for RNN-T
    use_triton       = True,  # set False if Triton unavailable
)
```

`OmegaConf.set_struct(cfg, False)` is called before writing `beam.boosting_tree`
because OmegaConf struct mode rejects new keys on an existing config object.

**`reset_decoding()`:**

Restores `self._base_decoding_cfg` via `change_decoding_strategy`. No-ops if
`_biasing_active` is `False` (safe to call unconditionally in `finally`).

---

### `pipeline.py` (modified)

**`submit()`** — new optional parameter `biasing_context: dict | None = None`.
Stored in the job record and forwarded to `_run`.

**`_run()`** — new steps after `_ensure_models()`:
```python
metrics["biasing"] = self._configure_biasing(language, biasing_context, job_dir)
biasing_applied = bool(metrics["biasing"] and metrics["biasing"].get("applied"))
```
And a `finally` block:
```python
finally:
    if biasing_applied:
        try: self.asr.reset_decoding()
        except Exception: pass   # never mask the job result
```

**`_configure_biasing(language, biasing_context, job_dir)`** — new private method:
1. Returns `None` if `biasing_context` is empty.
2. Returns `{"applied": False, "reason": "language_not_hindi"}` for non-Hindi.
3. Calls `build_key_phrases_file` and `asr.configure_biasing`.
4. On success returns:
   ```json
   {
     "applied": true,
     "dynamic_phrases": 4,
     "total_phrases": 53,
     "top_phrases": ["sushil kumar", "HDB Financial", "2026-01-05", "241509 rupees"],
     "key_phrases_file": "/tmp/nemotron-test/<job_id>/biasing/key_phrases.txt"
   }
   ```
5. On any exception returns `{"applied": False, "error": "ExcType: message"}` —
   the job continues unbiased.

The biasing metrics are included in the job result under `result.metrics.biasing`.

---

### `app.py` (modified)

Both `POST /api/jobs` and `POST /api/jobs/sample/{filename}` accept four new
optional form fields:

| Field | Type | Description |
|---|---|---|
| `name` | `str` (optional) | Debtor name for biasing |
| `institute_name` | `str` (optional) | Lender routing code or brand name |
| `total_due` | `str` (optional) | Amount owed (digits, optional `₹`/`rs.`) |
| `due_date` | `str` (optional) | Due date in any parseable format |

All four are bundled by `_biasing_context()` into a dict (skipping blank
values) and passed to `store.submit`. Omitting them all → `None` → no biasing,
exactly as before. Existing integrations are fully backwards-compatible.

---

### `eval_hindi_biasing.py` (new)

Batch A/B evaluation driver for the test CSVs.

```
python eval_hindi_biasing.py --csv Result_8.csv --limit 5
python eval_hindi_biasing.py --csv Result_8.csv --out my_results.csv
python eval_hindi_biasing.py --csv Result_7.csv --assume-hindi
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--csv` | `Result_8.csv` | Input CSV file |
| `--language` | `hi-IN` | Language tag for the pipeline |
| `--limit` | `0` (all) | Stop after N Hindi rows |
| `--out` | `biasing_eval_results.csv` | Output CSV path |
| `--assume-hindi` | off | Treat all rows as Hindi (for `Result_7.csv` which has no `language` column) |

For each Hindi row it:
1. Downloads the `cr_recording_url` mp3 over HTTPS.
2. Runs the pipeline **twice** — `biasing_context=None` (baseline) and
   `biasing_context=row` (biased).
3. Checks whether the debtor name and lender brand surface in each transcript
   (recall proxy — no reference transcripts exist).

**Output columns:**

```
idx, name, institute, brand, total_due, due_date,
biasing_applied, dynamic_phrases,
name_hit_baseline, name_hit_biased,
brand_hit_baseline, brand_hit_biased,
baseline_transcript, biased_transcript
```

Terminal summary:
```
name surfaced:  baseline 3 -> biased 7
brand surfaced: baseline 5 -> biased 9
```

---

### `probe_biasing.py` (new)

One-shot verification script. Run once on the GPU box before using the
integration to confirm the NeMo build and checkpoint support the boosting-tree
transducer path.

```
python probe_biasing.py
```

Prints:
```
model_class: EncDecRNNTBPEModel
has_joint(transducer): True
hybrid_aux_ctc: False
has_tokenizer: True
default_strategy: maes
BIASING APPLY: ok (malsd_batch + boosting_tree accepted)
reset_decoding: ok (restored maes)
```

If `BIASING APPLY: FAILED` is printed, the exception message indicates why
(missing Triton, no tokenizer, unsupported strategy, etc.) and biasing will
degrade gracefully to ITN-only in the pipeline.

---

## Adding a New Lender

If a new lender code appears in the CSV (e.g. `NEWBANK_RURAL`) and you want a
curated brand name rather than the title-case fallback:

1. Open `biasing_context.py`.
2. Add an entry to `INSTITUTE_BRANDS`:
   ```python
   "NEWBANK": "New Bank Finance",
   ```
3. Done. The suffix `_RURAL` is stripped automatically before lookup, so
   `NEWBANK_RURAL`, `NEWBANK_SPL`, etc. all resolve to the same brand.

No other files need to change. The phrase pack for every existing row is
unaffected.

---

## Tuning Biasing Strength

Two constants in `nemotron_streaming.py` control how strongly the boosting
tree overrides the acoustic model:

| Constant | Default | Effect |
|---|---|---|
| `BIASING_CONTEXT_SCORE` | `3.0` | Log-prob bonus per matched token arc. Higher → stronger pull toward biased phrases, but raises false-positive risk on similar-sounding words. |
| `BIASING_ALPHA` | `1.0` | Weight of the boosting tree relative to the main decoder score. Lower values blend it more gently. |

Start conservative (`context_score=2.0`, `alpha=0.8`) if you see the model
hallucinating biased phrases on audio that doesn't contain them.

---

## Relationship to Existing Context Biasing Module

`context_biasing/context_biasing.py` (the full CTC-WS runtime) is **not used**
by this integration. The reasons:

1. It loads a separate NeMo CTC model instance (additional GPU memory + load time).
2. Its decode knobs (`apply_context_biasing`, `context_file`) are NeMo CTC
   word-synchronous features that have no effect on an RNN-T `maes` decode.
3. `context_biasing/nemo_export.py` (which it imports) does not exist.

What *is* reused from `context_biasing/`:

| Module | Reused as |
|---|---|
| `context_assembler.py` | `parse_biasing_context`, `build_request_scoped_phrase_pack` |
| `variant_generator.py` | amount / date / name surface-form expansion |
| `hindi_transliteration.py` | Latin → Devanagari phonetic mapping |
| `phrase_ranker.py` | relevance ranking + deduplication |
| `phrases/hi.txt` | static Hindi financial-domain lexicon (138 canonical phrases) |

These five files import only from each other, so they work independently of
the missing `nemo_export.py` and the CTC-WS runtime.
