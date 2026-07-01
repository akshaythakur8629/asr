"""Deterministic Hinglish loan-domain term normalization."""

from __future__ import annotations

import re

from .contract import Span

RULE_ID = "lex.domain.loan_terms.v1"
_TERMS = (
    ("ई एम आई", "EMI"),
    ("इ एम आई", "EMI"),
    ("एम आई", "EMI"),
    ("ईएमआई", "EMI"),
    ("emi", "EMI"),
    ("सिबिल", "CIBIL"),
    ("सीबिल", "CIBIL"),
    ("सिविल", "CIBIL"),
    ("लोन", "loan"),
    ("पेमेंट", "payment"),
    ("पेमेन्ट", "payment"),
    ("ऐप", "app"),
    ("एप", "app"),
    ("लिंक", "link"),
    ("यूपीआई", "UPI"),
    ("यू पी आई", "UPI"),
    ("फोन पे", "PhonePe"),
    ("गूगल पे", "Google Pay"),
    ("पेटीएम", "Paytm"),
    ("सेटलमेंट", "settlement"),
    ("फोरक्लोजर", "foreclosure"),
    ("एन ओ सी", "NOC"),
    ("एनओसी", "NOC"),
    # Brand / model names common in collections calls.
    ("बजाज", "Bajaj"),
    ("पल्सर", "Pulsar"),
    # General English loanwords spoken in these calls but transcribed in
    # Devanagari. Multi-word phrases are listed so the longest-match policy
    # prefers them over their single-word constituents.
    ("गुड मॉर्निंग", "good morning"),
    ("गुड इवनिंग", "good evening"),
    ("गुड आफ्टरनून", "good afternoon"),
    ("पेनल्टी चार्जेस", "penalty charges"),
    ("पेनल्टी चार्ज", "penalty charge"),
    ("लास्ट डेट", "last date"),
    ("ड्यू डेट", "due date"),
    ("ड्यू डे", "due date"),
    ("पेनल्टी", "penalty"),
    ("चार्जेस", "charges"),
    ("चार्ज", "charge"),
    ("ऑटो", "auto"),
    ("अपडेट", "update"),
    ("कॉल", "call"),
    ("फ़ोन", "phone"),
    ("फोन", "phone"),
    ("लाइन", "line"),
    ("डेट", "date"),
    ("ड्यू", "due"),
    ("लेट", "late"),
    ("फ्री", "free"),
    ("थैंक यू", "thank you"),
    ("बाय", "bye"),
    ("बाय बाय", "bye bye"),
    # Bengali
    ("থ্যাঙ্ক ইউ", "thank you"),
    ("থ্যাংক ইউ", "thank you"),
    ("বাই", "bye"),
    ("বাই বাই", "bye bye"),
    # Telugu
    ("థాంక్యూ", "thank you"),
    ("బై", "bye"),
    ("బై బై", "bye bye"),
    # Tamil
    ("தேங்க்யூ", "thank you"),
    ("பை", "bye"),
    ("பை பை", "bye bye"),
    # Kannada
    ("ಥ್ಯಾಂಕ್ ಯೂ", "thank you"),
    ("ಥ್ಯಾಂಕ್ಯೂ", "thank you"),
    ("ಬೈ", "bye"),
    ("ಬೈ ಬೈ", "bye bye"),
    # Gujarati
    ("થેન્ક યુ", "thank you"),
    ("થેન્ક્યુ", "thank you"),
    ("બાય", "bye"),
    ("બાય બાય", "bye bye"),
    ("ओके", "OK"),
)

# Client banks / NBFCs / lenders (Devanagari, spaced-initialism, and romanized
# spellings) seen in CredResolve collections call flows.
_CLIENT_BANK_LENDER_TERMS = (
    # Known from your CredResolve call-flow/context examples
    ("मनी व्यू", "MoneyView"),
    ("मनीव्यू", "MoneyView"),
    ("moneyview", "MoneyView"),
    ("money view", "MoneyView"),

    ("टीवीएस", "TVS"),
    ("टी वी एस", "TVS"),
    ("टीवीएस क्रेडिट", "TVS Credit"),
    ("टी वी एस क्रेडिट", "TVS Credit"),
    ("tvs credit", "TVS Credit"),

    ("आई डी एफ सी", "IDFC FIRST Bank"),
    ("आईडीएफसी", "IDFC FIRST Bank"),
    ("आई डी एफ सी फर्स्ट", "IDFC FIRST Bank"),
    ("idfc", "IDFC FIRST Bank"),
    ("idfc first", "IDFC FIRST Bank"),

    ("एस एम एफ जी", "SMFG"),
    ("एसएमएफजी", "SMFG"),
    ("एस एम एफ जी इंडिया क्रेडिट", "SMFG India Credit"),
    ("एसएमएफजी इंडिया क्रेडिट", "SMFG India Credit"),
    ("smfg india credit", "SMFG India Credit"),

    ("एच डी एफ सी", "HDFC Bank"),
    ("एचडीएफसी", "HDFC Bank"),
    ("एच डी एफ सी बैंक", "HDFC Bank"),
    ("एचडीएफसी बैंक", "HDFC Bank"),
    ("hdfc", "HDFC Bank"),
    ("hdfc bank", "HDFC Bank"),

    ("एच डी बी", "HDB Financial Services"),
    ("एचडीबी", "HDB Financial Services"),
    ("hdb finance", "HDB Financial Services"),
    ("hdb financial", "HDB Financial Services"),

    ("यू ग्रो कैपिटल", "UGRO Capital"),
    ("यूग्रो कैपिटल", "UGRO Capital"),
    ("उग्रो कैपिटल", "UGRO Capital"),
    ("ugro capital", "UGRO Capital"),
    ("ugro captial", "UGRO Capital"),
    ("u gro capital", "UGRO Capital"),
    ("u grow capital", "UGRO Capital"),

    ("क्रेडिटबी", "KreditBee"),
    ("क्रेडिट बी", "KreditBee"),
    ("kreditbee", "KreditBee"),
    ("kredit bee", "KreditBee"),
    ("creditbee", "KreditBee"),
    ("credit bee", "KreditBee"),

    ("बाजाज फाइनेंस", "Bajaj Finance"),
    ("बजाज फाइनेंस", "Bajaj Finance"),
    ("bajaj finance", "Bajaj Finance"),
    ("बाजाज", "Bajaj"),
    ("bajaj", "Bajaj"),
)

_ALL_TERMS = _TERMS + _CLIENT_BANK_LENDER_TERMS
_WORD = r"A-Za-z0-9_\u0900-\u097f"
_PATTERNS = tuple(
    (re.compile(rf"(?<![{_WORD}]){re.escape(raw)}(?![{_WORD}])", re.IGNORECASE), canonical)
    for raw, canonical in sorted(_ALL_TERMS, key=lambda x: len(x[0]), reverse=True)
)


def detect_domain_terms(text: str) -> list[Span]:
    candidates = [
        (m.start(), m.end(), canonical)
        for pattern, canonical in _PATTERNS
        for m in pattern.finditer(text)
    ]
    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    selected: list[tuple[int, int, str]] = []
    for start, end, canonical in candidates:
        if not any(a < end and start < b for a, b, _ in selected):
            selected.append((start, end, canonical))
    return [
        Span(
            cls="domain_term",
            raw=text[a:b],
            canonical=c,
            rule_id=RULE_ID,
            conf=0.99,
            ambiguous=False,
            start=a,
            end=b,
            fallback_reason=None,
        )
        for a, b, c in sorted(selected)
    ]


__all__ = ["RULE_ID", "detect_domain_terms"]
