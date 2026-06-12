"""Normalize transcript JSONL files using an offline ITN backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..runtime.contract import SegmentResult
from ..runtime.offline_normalizer import OfflineComparison, normalize_offline_text


def _fields(r: SegmentResult) -> dict[str, Any]:
    return {
        "raw_text": r.raw_text,
        "canonical_text": r.canonical_text,
        "display_text": r.display_text,
        "spans": [s.model_dump(mode="json") for s in r.spans],
        "deferred": r.deferred,
        "lang": r.lang,
        "script": r.script,
        "itn_version": r.itn_version,
    }


def normalize_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lang: str = "hi",
    locale_policy: str = "india_default",
    backend: str = "custom",
) -> None:
    with (
        Path(input_path).open(encoding="utf-8") as source,
        Path(output_path).open("w", encoding="utf-8") as destination,
    ):
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"line {number}: text must be a string")
            result = normalize_offline_text(text, lang, locale_policy, backend=backend)
            output = dict(record)
            if isinstance(result, OfflineComparison):
                output.update(_fields(result.custom_result))
                output.update(
                    {
                        "custom_canonical_text": result.custom_result.canonical_text,
                        "nemo_canonical_text": result.nemo_result.canonical_text,
                        "nemo_display_text": result.nemo_result.display_text,
                        "nemo_spans": [s.model_dump(mode="json") for s in result.nemo_result.spans],
                        "changed_by_custom": result.changed_by_custom,
                        "changed_by_nemo": result.changed_by_nemo,
                        "outputs_equal": result.outputs_equal,
                    }
                )
            else:
                output.update(_fields(result))
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lang", default="hi")
    p.add_argument("--locale-policy", default="india_default")
    p.add_argument("--backend", choices=("custom", "nemo", "compare"), default="custom")
    a = p.parse_args()
    normalize_jsonl(
        a.input, a.output, lang=a.lang, locale_policy=a.locale_policy, backend=a.backend
    )


if __name__ == "__main__":
    main()
