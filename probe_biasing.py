"""Step 0 probe — run ONCE on the GPU box to confirm the biasing decode path.

Loads the Nemotron model and reports the facts the integration branches on:
  * model class + transducer type (RNN-T / TDT), and whether it's hybrid (aux_ctc)
  * presence of a tokenizer (the boosting tree tokenizes phrases with it)
  * that a malsd_batch + boosting_tree decode config applies without error

If the final "BIASING APPLY: ok" line prints, Case A is good and pipeline.py
biasing will work. Otherwise the printed exception tells you which fallback applies.

    python probe_biasing.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from nemotron_streaming import NemotronStreamingASR


def main() -> None:
    asr = NemotronStreamingASR(device="cuda:0")
    model = asr.model
    cfg = getattr(model, "cfg", None)
    print("model_class:", type(model).__name__)
    print("has_joint(transducer):", hasattr(model, "joint"))
    print("hybrid_aux_ctc:", hasattr(model, "aux_ctc") or getattr(cfg, "aux_ctc", None) is not None)
    print("has_tokenizer:", getattr(model, "tokenizer", None) is not None)
    print("default_strategy:", getattr(getattr(cfg, "decoding", None), "strategy", None))

    # Build a tiny key-phrases file and try to apply the boosting-tree decode.
    tmp = Path(tempfile.mkdtemp(prefix="probe-biasing-"))
    kp = tmp / "key_phrases.txt"
    kp.write_text("सुशील कुमार\nएचडीबी\nHDB Financial\n", encoding="utf-8")
    try:
        asr.configure_biasing(kp)
        print("BIASING APPLY: ok (malsd_batch + boosting_tree accepted)")
    except Exception as exc:  # noqa: BLE001 - we want the raw reason
        print(f"BIASING APPLY: FAILED -> {type(exc).__name__}: {exc}")
        print("Fallback: keep ITN-only for amounts/dates; names won't be biased on this build.")
        return
    finally:
        asr.reset_decoding()
    print("reset_decoding: ok (restored maes)")


if __name__ == "__main__":
    main()
