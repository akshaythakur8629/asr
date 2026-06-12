import json
from itn_service.tools.normalize_offline_transcript import normalize_jsonl


def test_cli_reads_and_writes_jsonl(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(
        json.dumps(
            {
                "call_id": "1",
                "segment_id": 1,
                "speaker": "ASSISTANT",
                "text": "ई एम आई पेंडिंग है",
                "extra": 7,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    normalize_jsonl(source, output, backend="custom")
    record = json.loads(output.read_text(encoding="utf-8"))
    assert (
        record["call_id"] == "1"
        and record["speaker"] == "ASSISTANT"
        and record["extra"] == 7
    )
    assert record["raw_text"] == "ई एम आई पेंडिंग है"
    assert "EMI" in record["canonical_text"]
    assert record["spans"][0]["rule_id"] == "lex.domain.loan_terms.v1"
