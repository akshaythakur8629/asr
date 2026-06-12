# Offline Hindi/Hinglish Transcript Normalization

Deterministic post-transcription normalization for the Hindi/Hinglish ASR pipeline. Runs after
transcription is complete (no streaming, no live path). No LLM, no WFST FARs required.

## Problem

The pipeline was producing under-normalized output:

| Issue | Example raw ASR output | Expected output |
| :--- | :--- | :--- |
| Currency spoken-form not converted | `छः हज़ार पाँच सौ रुपये` | `₹6,500` |
| English-origin words stay in Devanagari | `गुड इवनिंग`, `पेनल्टी चार्जेस` | `good evening`, `penalty charges` |
| Lender/bank names not normalized | `एच डी एफ सी बैंक`, `आई डी एफ सी फर्स्ट` | `HDFC Bank`, `IDFC FIRST Bank` |
| No human-readable output | Raw JSON / `role: text` lines | Markdown table |

**Root cause for currency:** The live-path normalizer (`make_wfst_classifier`) requires compiled
`.far` files (pynini/openfst toolchain). None are compiled on this host, so the pipeline silently
takes the `wfst_unavailable` branch and emits `canonical == raw` for money/date spans. The
pure-Python offline fallback hook (`enhance_unavailable_hindi_span`) was the correct extension
point but had not been wired for money spans.

## Architecture

```
normalize_offline_text(raw_text)
  └── normalize_segment(is_final=True, state=None)
        └── _with_domain_terms(make_wfst_classifier(...))
              ├── make_wfst_classifier  → wfst_unavailable spans (no FARs)
              ├── enhance_unavailable_hindi_span  ← extended for money/date/cardinal
              └── detect_domain_terms  ← extended with loanwords + lender names
```

All changes are additive and backward-compatible:
- `enhance_unavailable_hindi_span` only fires on spans with `wfst_unavailable` in
  `fallback_reason`. If FARs are ever compiled, WFST results take priority automatically.
- `detect_domain_terms` longest-match + overlap resolution was unchanged; only the term list grew.

## Changes

### 1. `itn_service/runtime/hindi_numerals.py` (new)

Pure-Python Hindi cardinal parser. No pynini dependency.

**Key exports:**

| Symbol | Purpose |
| :--- | :--- |
| `CARDINAL_VALUES: dict[str, int]` | Full 0–99 Devanagari spoken vocabulary + spelling variants (`पाँच/पांच`, `छह/छः`) + tiny romanized set (`nau`, `naur`, `tees`) |
| `RUPEE_CUES: frozenset[str]` | Currency surface forms: `{"रुपये", "रुपया", "रुपए", "रू", "₹"}` |
| `parse_hindi_cardinal(words) -> int | None` | Accumulate-with-scales algorithm; returns `None` on any unknown token (never guesses) |
| `format_inr(value) -> str` | `₹` + ICU-canonical Indian grouping via `grammars.common.indian_grouping` |

**Scale words handled:** `सौ` (×100), `हज़ार/हजार` (×1,000), `लाख` (×1,00,000), `करोड़` (×1,00,00,000), `अरब` (×1,00,00,00,000)

**Examples:**

```
["छः", "हज़ार", "पाँच", "सौ"]  →  6500      format_inr(6500)   →  "₹6,500"
["एक", "लाख", "पच्चीस", "हज़ार"] →  125000   format_inr(125000) →  "₹1,25,000"
["naur", "haazar", "tees"]      →  9030
["बजाज"]                         →  None      (unknown token — no guess)
```

**Algorithm (`parse_hindi_cardinal`):**
```
total = 0, current = 0
for each word:
  if unit (0–99)  → current += value
  if सौ           → current = (current or 1) * 100
  if हज़ार/लाख/… → total  += (current or 1) * scale; current = 0
  if unknown      → return None
return total + current
```

---

### 2. `itn_service/runtime/offline_hindi_fallback.py` (extended)

`enhance_unavailable_hindi_span` now handles three span classes:

| `span.cls` | Handler | Rule ID |
| :--- | :--- | :--- |
| `"date"` | `_parse_month_date` | `offline.hi.date_monthword.v1` |
| `"cardinal"` | `parse_hindi_cardinal` | `offline.hi.cardinal.v1` |
| `"money"` / `"amount"` / `"currency"` | `_parse_money` | `offline.hi.money.v1` |

`_parse_money` strips `RUPEE_CUES` tokens from the raw span, parses the remaining words with
`parse_hindi_cardinal`, and returns `format_inr(value)`. Gate acceptance verified:

| Gate condition | Required | Delivered |
| :--- | :--- | :--- |
| `classifier_min` | ≥ 0.90 | 0.99 (hard-coded on offline enhancement) |
| `asr_min` | ≥ 0.80 | 1.0 (offline path sets `asr_conf=1.0`) |
| Currency lex cue | `रुपये` in `span.raw` | ✓ (`_span_has_lex_cue` passes) |

On success the span's `fallback_reason` is cleared to `None`, `ambiguous` set to `False`.

**`_parse_cardinal` / `_parse_month_date`** now delegate to `parse_hindi_cardinal` from
`hindi_numerals` (previously used a local 0–31 dict). The date test `दस अप्रैल दो हज़ार छब्बीस → 10/04/26` continues to pass.

---

### 3. `itn_service/runtime/domain_terms.py` (extended)

#### English loanword additions to `_TERMS`

Multi-word phrases are listed before single-word constituents so the existing longest-match policy
picks the most-specific form:

| Devanagari | Canonical |
| :--- | :--- |
| `गुड मॉर्निंग`, `गुड इवनिंग`, `गुड आफ्टरनून` | `good morning`, `good evening`, `good afternoon` |
| `पेनल्टी चार्जेस`, `पेनल्टी चार्ज` | `penalty charges`, `penalty charge` |
| `लास्ट डेट` | `last date` |
| `ड्यू डेट`, `ड्यू डे` | `due date` |
| `पेनल्टी`, `चार्जेस`, `चार्ज` | `penalty`, `charges`, `charge` |
| `ऑटो`, `अपडेट`, `कॉल`, `फ़ोन/फोन`, `लाइन`, `डेट`, `ड्यू`, `लेट`, `फ्री`, `ओके` | `auto`, `update`, `call`, `phone`, `line`, `date`, `due`, `late`, `free`, `OK` |
| `बजाज`, `पल्सर` | `Bajaj`, `Pulsar` |

#### `_CLIENT_BANK_LENDER_TERMS` (new tuple)

Client bank/NBFC/lender names with Devanagari, spaced-initialism, and romanized spellings:

| Entity | Devanagari forms | Romanized forms |
| :--- | :--- | :--- |
| MoneyView | `मनी व्यू`, `मनीव्यू` | `moneyview`, `money view` |
| TVS / TVS Credit | `टीवीएस`, `टी वी एस`, `टीवीएस क्रेडिट`, `टी वी एस क्रेडिट` | `tvs credit` |
| IDFC FIRST Bank | `आई डी एफ सी`, `आईडीएफसी`, `आई डी एफ सी फर्स्ट` | `idfc`, `idfc first` |
| SMFG / SMFG India Credit | `एस एम एफ जी`, `एसएमएफजी`, `एस एम एफ जी इंडिया क्रेडिट`, `एसएमएफजी इंडिया क्रेडिट` | `smfg india credit` |
| HDFC Bank | `एच डी एफ सी`, `एचडीएफसी`, `एच डी एफ सी बैंक`, `एचडीएफसी बैंक` | `hdfc`, `hdfc bank` |
| HDB Financial Services | `एच डी बी`, `एचडीबी` | `hdb finance`, `hdb financial` |

`_ALL_TERMS = _TERMS + _CLIENT_BANK_LENDER_TERMS` feeds `_PATTERNS`, which is sorted by
descending raw-string length so multi-word phrases win over their prefixes (e.g.
`टीवीएस क्रेडिट → TVS Credit` wins over `टीवीएस → TVS`).

---

### 4. `transcript_render.py` (new, repo root)

Pure-function Markdown renderer for normalized pipeline output. No normalization logic — only presentation.

```python
render_markdown_transcript(turns: list[dict], *, text_field="canonical_text") -> str
```

**Output format:**
```markdown
### 📞 Call Transcript

| Time (s) | Speaker | Transcript |
| :--- | :--- | :--- |
| 0.00–5.23 | 👤 Customer | सर payment हो गए ₹6,500 |
| 5.50–8.10 | 🎧 Agent | ठीक है |
```

- Speaker mapped via `{"customer": "👤 Customer", "agent": "🎧 Agent"}`; unknown values title-cased.
- Empty turns (whitespace-only `canonical_text`) are skipped.
- `|` in transcript text escaped as `\|` to avoid breaking table structure.
- Overlap turns annotated with `· overlap` in the Time cell.
- `text_field` param lets callers substitute `"display_text"` or `"text"` if needed.

**Pipeline integration** (`pipeline.py` `_run`):
```python
result = {
    "turns": results,
    "transcript": "...",
    "markdown": render_markdown_transcript(results),   # ← added
    ...
}
```

---

## Tests

| File | Cases |
| :--- | :--- |
| `tests/offline/test_hindi_numerals.py` | units + scales, romanized variants, unknown rejection, `format_inr` Indian grouping |
| `tests/offline/test_offline_normalizer.py` | `छः हज़ार पाँच सौ रुपये → ₹6,500` end-to-end; span `fallback_reason` cleared; loanword rendering; existing date + cardinal tests unchanged |
| `tests/test_transcript_render.py` | table structure, emoji labels, pipe escaping, empty turn skipping |

Run all tests:
```bash
python3.11 -m pytest tests/ -v
```

Note: `.venv` is a Linux venv and cannot be activated on macOS. Use the system `python3.11` directly.

---

## Known Limitations

- ASR garbles not in the curated alias list remain verbatim (e.g. mis-recognized names). Fixing
  these requires an LLM or fuzzy matcher, both explicitly excluded.
- No prose Hindi→English translation column.
- Lexicon is curated, not exhaustive — new domain words are added by extending `_TERMS` or
  `_CLIENT_BANK_LENDER_TERMS`.
- Date normalization requires the pattern `<day-word> <month-word> [year-words]`; free-form date
  expressions (e.g. ordinals, AD/BC, relative dates) are left verbatim.
