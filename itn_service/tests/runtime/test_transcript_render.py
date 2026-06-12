from transcript_render import render_markdown_transcript


def _turn(speaker, start, end, canonical, **extra):
    return {"speaker": speaker, "start_sec": start, "end_sec": end,
            "canonical_text": canonical, "text": canonical, **extra}


def test_render_builds_table_with_roles_and_normalized_text():
    turns = [
        _turn("customer", 14.30, 25.58, "सर payment हो गए ₹6,500"),
        _turn("agent", 13.18, 14.09, "हाँ"),
    ]
    md = render_markdown_transcript(turns)
    assert "### 📞 Call Transcript" in md
    assert "| Time (s) | Speaker | Transcript |" in md
    assert "👤 Customer" in md and "🎧 Agent" in md
    assert "₹6,500" in md
    assert "14.30–25.58" in md


def test_render_skips_empty_and_escapes_pipes():
    turns = [
        _turn("customer", 0.0, 1.0, "   "),
        _turn("agent", 1.0, 2.0, "a | b"),
    ]
    md = render_markdown_transcript(turns)
    assert "a \\| b" in md
    # Only the non-empty agent row should appear as a data row.
    assert md.count("🎧 Agent") == 1
    assert "👤 Customer" not in md
