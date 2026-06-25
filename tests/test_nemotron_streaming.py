import contextlib
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from nemotron_model.model import (
    ATTENTION_CONTEXTS,
    DECODING_STRATEGY,
    MODEL_NAME,
    NemotronStreamingASR,
    _prompt_language,
    _prompt_transcribe_config,
)


class FakeTranscribeConfig:
    def __init__(self):
        self.use_lhotse = True
        self.batch_size = 4
        self.verbose = True
        self.target_lang = "auto"
        self.num_workers = None


class FakeCuda:
    def __init__(self):
        self.device = None

    def set_device(self, device):
        self.device = device


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()

    def inference_mode(self):
        return contextlib.nullcontext()


class FakeEncoder:
    def __init__(self):
        self.context = None

    def set_default_att_context_size(self, att_context_size):
        self.context = att_context_size


class FakeDecoding:
    def __init__(self):
        self.strip_lang_tags = None

    def set_strip_lang_tags(self, value):
        self.strip_lang_tags = value


class FakeModel:
    def __init__(self):
        self.encoder = FakeEncoder()
        self.decoding = FakeDecoding()
        self.prompt = None
        self.transcribe_audio = None
        self.transcribe_kwargs = None

    def get_transcribe_config(self):
        return FakeTranscribeConfig()

    def set_inference_prompt(self, target_lang):
        self.prompt = target_lang

    def transcribe(self, audio, **kwargs):
        self.transcribe_audio = audio
        self.transcribe_kwargs = kwargs
        return [SimpleNamespace(text="hello")]


class NemotronConfigurationTests(unittest.TestCase):
    def test_model_card_chunk_contexts(self):
        self.assertEqual(MODEL_NAME, "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.assertEqual(DECODING_STRATEGY, "maes")
        self.assertEqual(ATTENTION_CONTEXTS, {80: [56, 0], 160: [56, 1], 320: [56, 3], 560: [56, 6], 1120: [56, 13]})

    def test_prompt_language_normalizes_hindi_labels(self):
        self.assertEqual(_prompt_language(None), "hi-IN")
        self.assertEqual(_prompt_language("hindi"), "hi-IN")
        self.assertEqual(_prompt_language("hi_IN"), "hi-IN")
        self.assertEqual(_prompt_language("en-US"), "en-US")

    def test_prompt_transcribe_config_sets_target_and_disables_lhotse(self):
        cfg = _prompt_transcribe_config(FakeModel(), "hi-IN", batch_size=1, verbose=False)

        self.assertIsNotNone(cfg)
        self.assertFalse(cfg.use_lhotse)
        self.assertEqual(cfg.target_lang, "hi-IN")
        self.assertEqual(cfg.batch_size, 1)
        self.assertFalse(cfg.verbose)
        self.assertEqual(cfg.num_workers, 0)

    def test_transcribe_uses_prompt_override_config(self):
        model = FakeModel()
        asr = NemotronStreamingASR.__new__(NemotronStreamingASR)
        asr.torch = FakeTorch()
        asr.device = "cuda:0"
        asr.lock = threading.Lock()
        asr.model = model

        text = asr.transcribe(Path("clip.wav"), language="hindi")

        self.assertEqual(text, "hello")
        self.assertEqual(model.encoder.context, [-1, -1])
        self.assertEqual(model.prompt, "hi-IN")
        self.assertTrue(model.decoding.strip_lang_tags)
        self.assertEqual(model.transcribe_audio, ["clip.wav"])
        self.assertEqual(model.transcribe_kwargs["target_lang"], "hi-IN")
        cfg = model.transcribe_kwargs["override_config"]
        self.assertFalse(cfg.use_lhotse)
        self.assertEqual(cfg.target_lang, "hi-IN")


if __name__ == "__main__":
    unittest.main()
