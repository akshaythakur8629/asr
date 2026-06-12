"""Pure-Python spoken Hindi cardinal parsing for the offline (no-WFST) path.

The WFST grammars are the live-path authority for spoken numbers, but they require
compiled FARs and the native pynini/openfst toolchain. Offline we cannot rely on
those being present, so this module reproduces the subset the pipeline needs —
0-99 lexical cardinals plus the सौ / हज़ार / लाख / करोड़ scale words — as a small
deterministic parser. It has **no pynini dependency** and never guesses: any token
outside the known vocabulary makes :func:`parse_hindi_cardinal` return ``None`` so
the caller falls back to the verbatim text.

Currency rendering reuses :func:`grammars.common.indian_grouping.indian_grouping`
(also pure Python) so amounts get ICU-canonical Indian grouping (``6500`` → ``6,500``).
"""

from __future__ import annotations

from ..grammars.common.indian_grouping import indian_grouping

# ---------------------------------------------------------------------------
# Lexical inventory. Devanagari spellings mirror spoken_prefilter._NUMBER_WORDS
# so detected spans and parsed values stay in sync. A few common variants and a
# tiny romanized set (to keep the existing offline romanized-cardinal behaviour)
# are folded into the same maps; lookups casefold the token first.
# ---------------------------------------------------------------------------

CARDINAL_VALUES: dict[str, int] = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15, "पन्द्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18, "अट्ठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "इक्कीस": 21,
    "बाईस": 22,
    "तेईस": 23,
    "चौबीस": 24,
    "पच्चीस": 25,
    "छब्बीस": 26,
    "सत्ताईस": 27,
    "अट्ठाईस": 28,
    "उनतीस": 29,
    "तीस": 30,
    "इकतीस": 31,
    "बत्तीस": 32,
    "तैंतीस": 33,
    "चौंतीस": 34,
    "पैंतीस": 35,
    "छत्तीस": 36,
    "सैंतीस": 37,
    "अड़तीस": 38,
    "उनतालीस": 39,
    "चालीस": 40,
    "इकतालीस": 41,
    "बयालीस": 42,
    "तैंतालीस": 43,
    "चौंतालीस": 44,
    "पैंतालीस": 45,
    "छियालीस": 46,
    "सैंतालीस": 47,
    "अड़तालीस": 48,
    "उनचास": 49,
    "पचास": 50,
    "इक्यावन": 51,
    "बावन": 52,
    "तिरेपन": 53,
    "चौवन": 54,
    "पचपन": 55,
    "छप्पन": 56,
    "सत्तावन": 57,
    "अट्ठावन": 58,
    "उनसठ": 59,
    "साठ": 60,
    "इकसठ": 61,
    "बासठ": 62,
    "तिरेसठ": 63,
    "चौंसठ": 64,
    "पैंसठ": 65,
    "छयासठ": 66,
    "सरसठ": 67,
    "अड़सठ": 68,
    "उनहत्तर": 69,
    "सत्तर": 70,
    "इकहत्तर": 71,
    "बहत्तर": 72,
    "तिहत्तर": 73,
    "चौहत्तर": 74,
    "पचहत्तर": 75,
    "छिहत्तर": 76,
    "सतहत्तर": 77,
    "अठहत्तर": 78,
    "उन्यासी": 79,
    "अस्सी": 80,
    "इक्यासी": 81,
    "बयासी": 82,
    "तिरासी": 83,
    "चौरासी": 84,
    "पचासी": 85,
    "छियासी": 86,
    "सत्तासी": 87,
    "अठासी": 88,
    "नवासी": 89,
    "नब्बे": 90,
    "इक्यानवे": 91,
    "बानवे": 92,
    "तिरानवे": 93,
    "चौरानवे": 94,
    "पचानवे": 95,
    "छियानवे": 96,
    "सत्तानवे": 97,
    "अट्ठानवे": 98,
    "निन्यानवे": 99,
    # Tiny romanized set, preserving the existing offline behaviour.
    "nau": 9, "naur": 9, "tees": 30,
}

# सौ multiplies the running unit (पाँच सौ = 500); the larger scales flush the
# running value into the total (छः हज़ार पाँच सौ = 6000 + 500).
_HUNDRED: dict[str, int] = {"सौ": 100}
_LARGE_SCALES: dict[str, int] = {
    "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000,
    "करोड़": 10_000_000, "करोड": 10_000_000,
    "अरब": 1_000_000_000,
    # romanized
    "hazar": 1000, "haazar": 1000,
}

RUPEE_CUES: frozenset[str] = frozenset({"रुपये", "रुपया", "रुपए", "रू", "₹"})


def parse_hindi_cardinal(words: list[str]) -> int | None:
    """Convert a list of spoken Hindi number words to an integer.

    Returns ``None`` if ``words`` is empty or contains any token outside the
    known cardinal / scale vocabulary, so callers never normalise a guess.
    """
    if not words:
        return None
    total = 0
    current = 0
    saw_value = False
    for word in words:
        key = word.casefold()
        if key in CARDINAL_VALUES:
            current += CARDINAL_VALUES[key]
            saw_value = True
        elif key in _HUNDRED:
            current = (current or 1) * 100
            saw_value = True
        elif key in _LARGE_SCALES:
            total += (current or 1) * _LARGE_SCALES[key]
            current = 0
            saw_value = True
        else:
            return None
    if not saw_value:
        return None
    return total + current


def format_inr(value: int) -> str:
    """Render a non-negative integer rupee amount as ``₹`` + Indian grouping."""
    return "₹" + indian_grouping(str(value))


__all__ = ["CARDINAL_VALUES", "RUPEE_CUES", "format_inr", "parse_hindi_cardinal"]
