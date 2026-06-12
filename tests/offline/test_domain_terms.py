import re
from itn_service.runtime import domain_terms
from itn_service.runtime.domain_terms import RULE_ID, detect_domain_terms


def test_domain_term_span_contract_and_offsets():
    text = "कल फोन पे से पेमेंट करो"
    spans = detect_domain_terms(text)
    assert [(s.raw, s.canonical) for s in spans] == [
        ("फोन पे", "PhonePe"),
        ("पेमेंट", "payment"),
    ]
    for span in spans:
        assert text[span.start : span.end] == span.raw
        assert span.cls == "domain_term"
        assert span.rule_id == RULE_ID
        assert span.conf == 0.99
        assert span.fallback_reason is None


def test_domain_terms_prefer_longest_overlapping_match(monkeypatch):
    word = domain_terms._WORD
    monkeypatch.setattr(
        domain_terms,
        "_PATTERNS",
        (
            (re.compile(rf"(?<![{word}])फोन\ पे(?![{word}])"), "PhonePe"),
            (re.compile(rf"(?<![{word}])पे(?![{word}])"), "pay"),
        ),
    )
    spans = detect_domain_terms("फोन पे")
    assert [(s.raw, s.canonical) for s in spans] == [("फोन पे", "PhonePe")]


def test_lender_brand_terms_normalize_to_canonical_names():
    text = "यू ग्रो कैपिटल kredit bee और bajaj finance"
    spans = detect_domain_terms(text)
    assert [(s.raw, s.canonical) for s in spans] == [
        ("यू ग्रो कैपिटल", "UGRO Capital"),
        ("kredit bee", "KreditBee"),
        ("bajaj finance", "Bajaj Finance"),
    ]

