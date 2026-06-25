from utils.biasing_context import is_hindi, normalize_hindi_language


def test_common_hindi_csv_language_labels_normalize_to_nemo_prompt_key():
    for value in ("HINDI", "Hindi", "hi", "hi-IN", "hi_in", "hin", "हिंदी", "हिन्दी"):
        assert is_hindi(value)
        assert normalize_hindi_language(value) == "hi-IN"


def test_non_hindi_csv_language_labels_are_not_treated_as_hindi():
    for value in (None, "", "ENGLISH", "en-US", "MARATHI"):
        assert not is_hindi(value)
