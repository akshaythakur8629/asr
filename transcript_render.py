"""Render normalized speaker turns as a human-readable Markdown transcript.

This is an offline, post-transcription presentation step: it consumes the
already-normalized turns produced by the pipeline (``canonical_text`` per turn)
and emits a clean Markdown table. It performs no normalization itself.
"""
from __future__ import annotations

from typing import Any

_SPEAKER_LABELS = {"customer": "👤 Customer", "agent": "🎧 Agent"}


def _label(speaker: str) -> str:
    if speaker in _SPEAKER_LABELS:
        return _SPEAKER_LABELS[speaker]
    return speaker.title() if speaker else "Speaker"


def _cell(text: str) -> str:
    """Make text safe for a single Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def render_markdown_transcript(
    turns: list[dict[str, Any]], *, text_field: str = "canonical_text"
) -> str:
    """Build a Markdown table: Time (s) | Speaker | Transcript.

    Uses each turn's normalized text (``canonical_text`` by default, falling
    back to ``text``). Empty turns are skipped.
    """
    header = (
        "### 📞 Call Transcript\n\n"
        "| Time (s) | Speaker | Transcript |\n"
        "| :--- | :--- | :--- |\n"
    )
    rows: list[str] = []
    for turn in turns:
        text = turn.get(text_field) or turn.get("text") or ""
        if not str(text).strip():
            continue
        start = float(turn.get("start_sec", 0.0))
        end = float(turn.get("end_sec", 0.0))
        overlap = " · overlap" if turn.get("overlap_flag") else ""
        rows.append(
            f"| {start:.2f}–{end:.2f}{overlap} "
            f"| {_label(str(turn.get('speaker', '')))} "
            f"| {_cell(text)} |"
        )
    return header + "\n".join(rows) + ("\n" if rows else "")


__all__ = ["render_markdown_transcript"]
