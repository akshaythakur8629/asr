"""Batch driver: A/B test per-row context biasing on the Hindi rows of a CSV.

Reads a Result_*.csv, keeps the HINDI rows (Result_8.csv has a `language`
column; Result_7.csv does not, so pass --assume-hindi for it), downloads each
recording, and runs every row through the pipeline TWICE — baseline (no biasing)
vs biased (name + institute_name + total_due + due_date) — both with ITN — then
writes a side-by-side comparison plus a name/brand recall proxy.

    python eval_hindi_biasing.py --csv Result_8.csv --limit 5
    python eval_hindi_biasing.py --csv Result_8.csv --out eval_results.csv

No reference transcripts exist, so the "hit" columns are recall proxies
(did the row's name / brand surface in the transcript), not WER.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from .biasing_context import institute_to_brand, is_hindi, normalize_hindi_language
from context_biasing.variant_generator import generate_general_variants
from .pipeline import JobStore


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    return dest


def run_job(store: JobStore, audio: Path, language: str, biasing_context) -> dict:
    job = store.submit(audio, audio.name, language=language, biasing_context=biasing_context)
    job_id = job["id"]
    while True:
        time.sleep(2)
        state = store.public(job_id)
        if state and state["status"] in {"complete", "failed"}:
            if state["status"] == "failed":
                detail = state.get("traceback") or state.get("error") or "unknown pipeline error"
                raise RuntimeError(f"Job {job_id} failed:\n{detail}")
            return state


def surfaced(text: str, value: str) -> bool:
    """True if `value` (or any Hindi/Latin variant of it) appears in `text`."""
    if not value or not text:
        return False
    low = text.lower()
    return any(v.lower() in low for v in generate_general_variants(value, language="hi"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Result_8.csv")
    ap.add_argument("--language", default="hi-IN")
    ap.add_argument("--limit", type=int, default=0, help="0 = all Hindi rows")
    ap.add_argument("--out", default="biasing_eval_results.csv")
    ap.add_argument("--assume-hindi", action="store_true", help="treat every row as Hindi (no language column)")
    args = ap.parse_args()

    requested_language = normalize_hindi_language(args.language) or "hi-IN"
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    hindi = [r for r in rows if args.assume_hindi or is_hindi(r.get("language"))]
    if args.limit:
        hindi = hindi[: args.limit]
    if not hindi:
        sys.exit("No HINDI rows found (use --assume-hindi for a CSV without a language column).")
    print(f"{len(hindi)} Hindi rows to evaluate (baseline + biased each).")

    store = JobStore()
    work = Path(tempfile.mkdtemp(prefix="hindi-eval-"))
    out_rows = []
    for i, row in enumerate(hindi, 1):
        name, inst = (row.get("name") or "").strip(), (row.get("institute_name") or "").strip()
        brand = institute_to_brand(inst)
        print(f"[{i}/{len(hindi)}] {name} / {inst} -> {brand}")
        try:
            audio = download(row["cr_recording_url"], work / f"row_{i:03d}.mp3")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Row {i} recording download failed: {exc}") from exc

        base = run_job(store, audio, requested_language, None)
        biased = run_job(store, audio, requested_language, row)
        base_txt = (base.get("result") or {}).get("transcript", "") if base["status"] == "complete" else f"<{base.get('error')}>"
        biased_res = biased.get("result") or {}
        biased_txt = biased_res.get("transcript", "") if biased["status"] == "complete" else f"<{biased.get('error')}>"
        bias_meta = (biased_res.get("metrics") or {}).get("biasing") or {}

        out_rows.append({
            "idx": i, "cr_recording_url": row.get("cr_recording_url", ""),
            "name": name, "institute": inst, "brand": brand,
            "total_due": row.get("total_due", ""), "due_date": (row.get("due_date") or "").split(" ")[0],
            "biasing_applied": bias_meta.get("applied"), "dynamic_phrases": bias_meta.get("dynamic_phrases"),
            "name_hit_baseline": surfaced(base_txt, name), "name_hit_biased": surfaced(biased_txt, name),
            "brand_hit_baseline": surfaced(base_txt, brand), "brand_hit_biased": surfaced(biased_txt, brand),
            "baseline_transcript": base_txt, "biased_transcript": biased_txt,
        })

    if out_rows:
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            writer.writeheader(); writer.writerows(out_rows)
        nb = sum(r["name_hit_baseline"] for r in out_rows); nx = sum(r["name_hit_biased"] for r in out_rows)
        bb = sum(r["brand_hit_baseline"] for r in out_rows); bx = sum(r["brand_hit_biased"] for r in out_rows)
        print(f"\nWrote {args.out} ({len(out_rows)} rows)")
        print(f"name surfaced:  baseline {nb} -> biased {nx}")
        print(f"brand surfaced: baseline {bb} -> biased {bx}")


if __name__ == "__main__":
    main()