"""Offline, final-segment ITN normalization interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from .confidence_gate import ThresholdTable, load_thresholds
from .contract import SegmentResult, Span, Token
from .domain_terms import detect_domain_terms
from .locale_policy import load_locale_policy
from .nemo_itn_adapter import NemoItnAdapter
from .offline_hindi_fallback import detect_romanized_cardinals, enhance_unavailable_hindi_span
from .normalizer import Classifier, normalize_segment
from .wfst_classifier import make_wfst_classifier


@dataclass(frozen=True)
class OfflineComparison:
    raw_text: str
    custom_result: SegmentResult
    nemo_result: SegmentResult
    changed_by_custom: bool
    changed_by_nemo: bool
    outputs_equal: bool


@cache
def _thresholds() -> ThresholdTable:
    return load_thresholds()


@cache
def _production_classifier(locale_policy: str) -> Classifier:
    return _with_domain_terms(make_wfst_classifier(load_locale_policy().for_tenant(locale_policy)))


def _overlaps(a: Span, b: Span) -> bool:
    return (
        a.start is not None
        and a.end is not None
        and b.start is not None
        and b.end is not None
        and a.start < b.end
        and b.start < a.end
    )


def _with_domain_terms(classifier: Classifier) -> Classifier:
    def classify(text: str, lang: str) -> list[Span]:
        existing = [enhance_unavailable_hindi_span(span) if lang == "hi" else span for span in classifier(text, lang)]
        additions = [*detect_domain_terms(text)]
        if lang == "hi":
            additions.extend(detect_romanized_cardinals(text))
        result = existing + [span for span in additions if not any(_overlaps(span, old) for old in existing)]
        return sorted(result, key=lambda span: span.start if span.start is not None else len(text))

    return classify


def normalize_offline_text(
    text: str,
    lang_hint: str = "hi",
    locale_policy: str = "india_default",
    tokens: list[Token] | None = None,
    backend: str = "custom",
    *,
    thresholds: ThresholdTable | None = None,
    classifier: Classifier | None = None,
    nemo_adapter: NemoItnAdapter | None = None,
) -> SegmentResult | OfflineComparison:
    if backend not in {"custom", "nemo", "compare"}:
        raise ValueError(f"unsupported offline ITN backend: {backend!r}")
    if backend in {"custom", "compare"}:
        try:
            custom = normalize_segment(
                raw_text=text,
                tokens=tokens or [],
                is_final=True,
                state=None,
                lang_hint=lang_hint,
                locale_policy=locale_policy,
                thresholds=thresholds or _thresholds(),
                classifier=_with_domain_terms(classifier)
                if classifier
                else _production_classifier(locale_policy),
            )
        except Exception:
            custom = _passthrough(text, lang_hint)
        if backend == "custom":
            return custom
    nemo = (nemo_adapter or NemoItnAdapter()).normalize_text(text, lang_hint)
    if backend == "nemo":
        return nemo
    return OfflineComparison(
        text,
        custom,
        nemo,
        custom.canonical_text != text,
        nemo.canonical_text != text,
        custom.canonical_text == nemo.canonical_text,
    )


def _passthrough(text: str, lang: str) -> SegmentResult:
    return SegmentResult(
        raw_text=text,
        canonical_text=text,
        display_text=text,
        spans=[],
        deferred=True,
        lang=lang,
        script="Common",
    )


def normalize_offline_segments(
    texts: list[str],
    lang_hint: str = "hi",
    locale_policy: str = "india_default",
    tokens: list[list[Token]] | None = None,
    backend: str = "custom",
    *,
    thresholds: ThresholdTable | None = None,
    classifier: Classifier | None = None,
    nemo_adapter: NemoItnAdapter | None = None,
) -> list[SegmentResult | OfflineComparison]:
    token_lists = tokens if tokens is not None else [[] for _ in texts]
    if len(token_lists) != len(texts):
        raise ValueError("tokens must contain one token list per text")
    return [
        normalize_offline_text(
            t,
            lang_hint,
            locale_policy,
            ts,
            backend,
            thresholds=thresholds,
            classifier=classifier,
            nemo_adapter=nemo_adapter,
        )
        for t, ts in zip(texts, token_lists, strict=True)
    ]


__all__ = ["OfflineComparison", "normalize_offline_segments", "normalize_offline_text"]
