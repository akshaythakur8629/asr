from itn_service.runtime.contract import SegmentResult
from itn_service.runtime.offline_normalizer import (
    OfflineComparison,
    normalize_offline_text,
)


class FakeNemo:
    def normalize_text(self, text, lang="hi"):
        return SegmentResult(
            raw_text=text,
            canonical_text="NEMO",
            display_text="NEMO",
            spans=[],
            deferred=False,
            lang=lang,
            script="Devanagari",
        )


def test_custom_backend_returns_segment_result_and_preserves_raw():
    raw = "  ई एम आई पेंडिंग है  "
    result = normalize_offline_text(raw)
    assert isinstance(result, SegmentResult)
    assert result.raw_text == raw
    assert "EMI" in result.canonical_text


def test_custom_backend_calls_final_path_without_stream_state(monkeypatch):
    calls = {}

    def fake_normalize_segment(*args, **kwargs):
        calls.update(kwargs)
        return SegmentResult(
            raw_text=kwargs["raw_text"],
            canonical_text=kwargs["raw_text"],
            display_text=kwargs["raw_text"],
            spans=[],
            deferred=False,
            lang="hi",
            script="Devanagari",
        )

    monkeypatch.setattr(
        "itn_service.runtime.offline_normalizer.normalize_segment",
        fake_normalize_segment,
    )
    normalize_offline_text("text", classifier=lambda text, lang: [])
    assert calls["is_final"] is True
    assert calls["state"] is None
    assert calls["tokens"] == []


def test_expected_domain_terms_are_normalized():
    cases = (
        ("ई एम आई पेंडिंग है", "EMI"),
        ("सिबिल खराब होगा", "CIBIL"),
        ("फोन पे से पेमेंट कर दो", "PhonePe"),
        ("फोन पे से पेमेंट कर दो", "payment"),
        ("यू ग्रो कैपिटल से कॉल है", "UGRO Capital"),
        ("क्रेडिट बी ऐप में चेक करो", "KreditBee"),
        ("bajaj finance loan है", "Bajaj Finance"),
    )
    for raw, expected in cases:
        assert expected in normalize_offline_text(raw).canonical_text


def test_compare_backend_returns_both_outputs():
    result = normalize_offline_text(
        "ई एम आई", backend="compare", nemo_adapter=FakeNemo()
    )
    assert isinstance(result, OfflineComparison)
    assert result.custom_result.canonical_text == "EMI"
    assert result.nemo_result.canonical_text == "NEMO"
    assert result.outputs_equal is False


def test_offline_hindi_fallback_normalizes_reported_date_and_cardinals():
    assert normalize_offline_text("दस अप्रैल दो हज़ार छब्बीस").canonical_text == "10/04/26"
    assert normalize_offline_text("नौ हजार तीस").canonical_text == "9030"
    assert normalize_offline_text("naur haazar tees").canonical_text == "9030"


def test_offline_money_normalized_without_wfst():
    # The reported gap: spoken rupee amount must render as ₹ + Indian grouping
    # even though no WFST FARs are compiled.
    result = normalize_offline_text("छः हज़ार पाँच सौ रुपये payment हो गए")
    assert "₹6,500" in result.canonical_text
    assert "रुपये" not in result.canonical_text
    money = [s for s in result.spans if s.cls == "money"]
    assert money and money[0].canonical == "₹6,500"
    assert money[0].fallback_reason is None


def test_offline_english_loanwords_render_in_english():
    raw = "गुड इवनिंग बजाज ऑटो वाली लास्ट डेट है पेनल्टी चार्जेस लगेंगे"
    canonical = normalize_offline_text(raw).canonical_text
    for expected in ("good evening", "Bajaj", "auto", "last date", "penalty charges"):
        assert expected in canonical
