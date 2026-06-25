"""Per-row dynamic context-biasing phrase-pack assembly for Hindi calls.

Reuses the phrase-assembly half of ``context_biasing/`` —
``parse_biasing_context`` + ``build_request_scoped_phrase_pack`` — which depend
only on ``phrase_ranker`` / ``variant_generator`` (NOT on the missing
``nemo_export.py`` or the CTC-WS runtime). It turns a CSV row's
``{name, institute_name, total_due, due_date}`` into a NeMo boosting-tree
key-phrases file (one phrase per line) that the transducer decode can consume.

Everything here is rebuilt per row at runtime, so adding a new CSV row needs no
code change: names/amounts/dates are read straight off the row, and unknown
institute codes fall back to a title-cased brand (see ``institute_to_brand``).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from context_biasing.context_assembler import (
    AssembledPhrasePack,
    build_request_scoped_phrase_pack,
    parse_biasing_context,
)
from context_biasing.hindi_transliteration import LETTER_NAMES

# Devanagari letter-names ("पी","एन","एस"…) the acronym transliterator emits, e.g.
# "SMFG" -> "एस एम एफ जी". A phrase made only of these (or single Latin letters) is a
# spell-out, not a real phrase, and is a runaway seed under boosting — see _flatten_pack.
_SPELLOUT_TOKENS = frozenset(LETTER_NAMES.values())

PHRASES_DIR = Path(__file__).parent / "context_biasing" / "phrases"

# Routing suffixes appended to lender codes that aren't part of the spoken brand.
_INSTITUTE_SUFFIXES = ("_SPL", "_SOUTH", "_NACL", "_PL", "_RURAL", "_AUTO")

# Institute routing code -> spoken brand name. Unmapped codes fall back to
# title-casing in ``institute_to_brand``, so a brand-new lender still biases
# (just with a title-cased pronunciation instead of a curated one).
INSTITUTE_BRANDS: dict[str, str] = {
    "UGRO_CAPITAL": "Ugro Capital",
    "MOBIKWIK": "MobiKwik",
    "MONEYVIEW": "Money View",
    "CASHE": "CASHe",
    "KREDITBEE": "KreditBee",
    "BOBCARD": "BOB Card",
    "HDBFS": "HDB Financial",
    "PAYME": "PayMe",
    "IARC": "IARC",
    "PAHAL_FINANCE": "Pahal Finance",
    "BAJAJ": "Bajaj",
    "SMFG": "SMFG",
    "AGRIM": "Agrim",
}


def institute_to_brand(code: str) -> str:
    """Map an institute routing code to its spoken brand name."""
    raw = (code or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper in INSTITUTE_BRANDS:
        return INSTITUTE_BRANDS[upper]
    stripped = upper
    for suffix in _INSTITUTE_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    if stripped in INSTITUTE_BRANDS:
        return INSTITUTE_BRANDS[stripped]
    return stripped.replace("_", " ").title()


def _clean_amount(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text or text in {"0", "0.0", "0.00"}:
        return ""
    return text


def _clean_date(value: Any) -> str:
    # CSV dates look like "2026-01-05 00:00:00.000000"; keep just the date part.
    text = str(value if value is not None else "").strip()
    return text.split(" ", 1)[0] if text else ""


_HINDI_LANGUAGE_MARKERS = {"hi", "hi-in", "hi_in", "hi in", "hindi", "hin", "हिंदी", "हिन्दी"}


def normalize_hindi_language(language: str | None) -> str:
    """Return the NeMo Hindi prompt key for common CSV/UI language labels."""
    value = str(language or "").strip()
    marker = value.lower().replace("_", "-")
    if marker in _HINDI_LANGUAGE_MARKERS or marker.split("-", 1)[0] == "hi":
        return "hi-IN"
    return value


def is_hindi(language: str | None) -> bool:
    return normalize_hindi_language(language) == "hi-IN"


def row_to_biasing_context(row: dict[str, Any]) -> dict[str, Any]:
    """Map CSV columns onto context_biasing field names."""
    ctx: dict[str, Any] = {}
    name = str(row.get("name") or "").strip()
    if name:
        ctx["debtor_name"] = name
    brand = institute_to_brand(str(row.get("institute_name") or ""))
    if brand:
        ctx["lender"] = brand
    amount = _clean_amount(row.get("total_due"))
    if amount:
        # Collection calls speak the figure in rupees; the "rupees" hint makes
        # variant_generator emit Devanagari + word + currency forms.
        ctx["amounts"] = [f"{amount} rupees"]
    date = _clean_date(row.get("due_date"))
    if date:
        ctx["dates"] = [date]
    return ctx


def _flatten_pack(pack: AssembledPhrasePack) -> list[str]:
    """``pack.lines`` are underscore-joined variant groups; the NeMo boosting
    tree wants one phrase per line, so split every group into its variants."""
    phrases: list[str] = []
    seen: set[str] = set()
    for line in pack.lines:
        for variant in line.split("_"):
            phrase = variant.strip()
            key = phrase.lower()
            if not phrase or key in seen:
                continue
            # Skip spell-out variants (Latin "p a n" / "n a c h", or their Devanagari
            # letter-name forms "पी ए एन" / "एस एम एफ जी"): short, generic letter
            # sequences that match almost any frame cheaply, so a boosted decode falls
            # into a repeating P-A-N-P-A-N loop on them. We key off the letter-name set
            # rather than token length so real number words (e.g. "दो सौ") survive.
            tokens = phrase.split()
            if len(tokens) > 1 and all(
                len(t) == 1 or t in _SPELLOUT_TOKENS for t in tokens
            ):
                continue
            seen.add(key)
            phrases.append(phrase)
    return phrases


def build_key_phrases_file(
    row: dict[str, Any],
    *,
    language: str,
    out_dir: Path | None = None,
    max_dynamic_phrases: int = 32,
    include_static_lexicon: bool = False,
) -> tuple[Path, AssembledPhrasePack] | None:
    """Build a per-row boosting-tree key-phrases file.

    Returns ``(path, pack)`` or ``None`` when biasing does not apply (non-Hindi
    language, or the row carries no usable fields).
    """
    if not is_hindi(language):
        return None
    context = parse_biasing_context(row_to_biasing_context(row))
    if not context.provided_fields:
        return None
    # The static lexicon (hi.txt) carries every lender's brand plus legal boilerplate
    # ("smfg india credit", "ugro capital", "recovery agent", "no objection certificate"),
    # so attaching it to a single debtor's pack boosts brands this call never mentions and
    # the decode regurgitates them (a Money View call comes back full of "SMFG"/"UGRO").
    # Default off: bias only on THIS row's name/lender/amount/date. Opt back in per-call.
    base = (PHRASES_DIR / "hi.txt") if include_static_lexicon else None
    pack = build_request_scoped_phrase_pack(
        context=context,
        base_phrase_file=str(base) if base and base.exists() else None,
        max_dynamic_phrases=max_dynamic_phrases,
        language="hi",
    )
    if pack is None:
        return None
    phrases = _flatten_pack(pack)
    if not phrases:
        return None
    target_dir = out_dir or Path(tempfile.mkdtemp(prefix="biasing-"))
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "key_phrases.txt"
    path.write_text("\n".join(phrases) + "\n", encoding="utf-8")
    return path, pack