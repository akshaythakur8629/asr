"""Optional adapter for NeMo's alignment-preserving inverse normalizer."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

from .contract import SegmentResult
from .script_router import detect_script

AlignmentPreservingInverseNormalizer: Any | None = None
_NEMO_IMPORT_ERROR: Exception | None = None


class NemoUnavailableError(RuntimeError):
    """Raised only when the optional NeMo backend cannot be imported."""


def _load_nemo_normalizer_class() -> Any:
    global AlignmentPreservingInverseNormalizer, _NEMO_IMPORT_ERROR
    if AlignmentPreservingInverseNormalizer is not None:
        return AlignmentPreservingInverseNormalizer
    if _NEMO_IMPORT_ERROR is not None:
        raise NemoUnavailableError(
            "NeMo ITN backend is unavailable; install NeMo ASR and "
            "nemo_text_processing dependencies"
        ) from _NEMO_IMPORT_ERROR
    try:
        from nemo.collections.asr.inference.itn.inverse_normalizer import (
            AlignmentPreservingInverseNormalizer as NormalizerClass,
        )
    except Exception as exc:
        _NEMO_IMPORT_ERROR = exc
        raise NemoUnavailableError(
            "NeMo ITN backend is unavailable; install NeMo ASR and "
            "nemo_text_processing dependencies"
        ) from exc
    AlignmentPreservingInverseNormalizer = NormalizerClass
    return NormalizerClass


class NemoItnAdapter:
    _instances: ClassVar[dict[tuple[str, str], Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __init__(self, cache_dir: str | Path = "/tmp/nemo_itn_cache") -> None:
        self.cache_dir = str(cache_dir)

    def _normalizer_for(self, lang: str) -> Any:
        key = (lang, self.cache_dir)
        with self._lock:
            if key not in self._instances:
                self._instances[key] = _load_nemo_normalizer_class()(
                    lang=lang, cache_dir=self.cache_dir, overwrite_cache=False
                )
            return self._instances[key]

    def normalize_text(self, text: str, lang: str = "hi") -> SegmentResult:
        try:
            outputs = self._normalizer_for(lang).inverse_normalize_list([text], params={})
            if len(outputs) != 1 or not isinstance(outputs[0], str):
                raise ValueError("NeMo ITN returned an invalid result")
            normalized = outputs[0]
        except NemoUnavailableError:
            raise
        except Exception:
            return _passthrough(text, lang)
        return SegmentResult(
            raw_text=text,
            canonical_text=normalized,
            display_text=normalized,
            spans=[],
            deferred=False,
            lang=lang,
            script=_detect_script(text),
        )

    def normalize_list(self, texts: list[str], lang: str = "hi") -> list[SegmentResult]:
        return [self.normalize_text(text, lang) for text in texts]


def _detect_script(text: str) -> str:
    try:
        return detect_script(text)
    except Exception:
        return "Common"


def _passthrough(text: str, lang: str) -> SegmentResult:
    return SegmentResult(
        raw_text=text,
        canonical_text=text,
        display_text=text,
        spans=[],
        deferred=True,
        lang=lang,
        script=_detect_script(text),
    )


__all__ = ["NemoItnAdapter", "NemoUnavailableError"]
