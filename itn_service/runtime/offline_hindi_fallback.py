"""Narrow deterministic Hindi fallbacks for offline mode when FARs are unavailable."""
from __future__ import annotations

import re
from .contract import Span

_NUMBERS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18,
    "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31,
}
_ROMAN_NUMBERS = {"nau": 9, "naur": 9, "tees": 30}
_MONTHS = {"जनवरी": 1, "फरवरी": 2, "फ़रवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5,
           "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9,
           "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12}
_SPACE_RE = re.compile(r"\s+")


def enhance_unavailable_hindi_span(span: Span) -> Span:
    """Replace only a classified Hindi span whose WFST is unavailable."""
    canonical = None
    rule_id = span.rule_id
    if span.cls == "date":
        canonical = _parse_month_date(span.raw)
        rule_id = "offline.hi.date_monthword.v1"
    elif span.cls == "cardinal":
        canonical = _parse_cardinal(span.raw)
        rule_id = "offline.hi.cardinal.v1"
    if canonical is None or (span.fallback_reason and "wfst_unavailable" not in span.fallback_reason):
        return span
    return span.model_copy(update={"canonical": str(canonical), "rule_id": rule_id,
                                   "ambiguous": False, "fallback_reason": None, "conf": 0.99})


def detect_romanized_cardinals(text: str) -> list[Span]:
    """Detect a deliberately tiny romanized Hindi cardinal vocabulary."""
    pattern = re.compile(r"(?<![A-Za-z])(?:nau|naur)\s+(?:hazar|haazar)\s+tees(?![A-Za-z])", re.I)
    return [Span(cls="cardinal", raw=m.group(0), canonical=str(_parse_cardinal(m.group(0))),
                 rule_id="offline.hi.cardinal.romanized.v1", conf=0.99, ambiguous=False,
                 start=m.start(), end=m.end(), fallback_reason=None) for m in pattern.finditer(text)]


def _parse_month_date(raw: str) -> str | None:
    words = _SPACE_RE.split(raw.strip())
    if len(words) < 2 or words[0] not in _NUMBERS or words[1] not in _MONTHS:
        return None
    day = _NUMBERS[words[0]]
    if not 1 <= day <= 31:
        return None
    result = f"{day:02d}/{_MONTHS[words[1]]:02d}"
    if len(words) == 2:
        return result
    year = _parse_cardinal(" ".join(words[2:]))
    if year is None or not 1900 <= year <= 2099:
        return None
    return f"{result}/{year % 100:02d}"


def _parse_cardinal(raw: str) -> int | None:
    words = _SPACE_RE.split(raw.strip().lower())
    if not words:
        return None
    values = _ROMAN_NUMBERS if all(word.isascii() for word in words) else _NUMBERS
    thousand_words = {"हजार", "हज़ार", "hazar", "haazar"}
    if len(words) == 1:
        return values.get(words[0])
    if len(words) in {2, 3} and words[1] in thousand_words and words[0] in values:
        tail = 0 if len(words) == 2 else values.get(words[2])
        return None if tail is None else values[words[0]] * 1000 + tail
    return None


__all__ = ["detect_romanized_cardinals", "enhance_unavailable_hindi_span"]
