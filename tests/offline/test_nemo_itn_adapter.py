import pytest
from itn_service.runtime import nemo_itn_adapter
from itn_service.runtime.nemo_itn_adapter import NemoItnAdapter
from itn_service.runtime.offline_normalizer import normalize_offline_text


def reset(monkeypatch, cls=None, error=None):
    monkeypatch.setattr(nemo_itn_adapter, "AlignmentPreservingInverseNormalizer", cls)
    monkeypatch.setattr(nemo_itn_adapter, "_NEMO_IMPORT_ERROR", error)
    NemoItnAdapter._instances.clear()


def test_missing_nemo_does_not_break_custom_backend(monkeypatch):
    reset(monkeypatch, error=ImportError("missing"))
    assert normalize_offline_text("ई एम आई").canonical_text == "EMI"
    with pytest.raises(RuntimeError, match="NeMo ITN backend is unavailable"):
        normalize_offline_text("ई एम आई", backend="nemo")


def test_nemo_runtime_failure_returns_deferred_passthrough(monkeypatch):
    class Broken:
        def __init__(self, **kwargs):
            pass

        def inverse_normalize_list(self, texts, params):
            raise RuntimeError("failed")

    reset(monkeypatch, cls=Broken)
    result = NemoItnAdapter().normalize_text("raw", "hi")
    assert result.raw_text == result.canonical_text == result.display_text == "raw"
    assert result.deferred is True


def test_nemo_success_and_instance_cache(monkeypatch):
    made = []

    class Fake:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def inverse_normalize_list(self, texts, params):
            return ["normalized" for _ in texts]

    reset(monkeypatch, cls=Fake)
    adapter = NemoItnAdapter("/tmp/test-nemo-cache")
    assert adapter.normalize_text("one").canonical_text == "normalized"
    assert adapter.normalize_text("two").canonical_text == "normalized"
    assert len(made) == 1
