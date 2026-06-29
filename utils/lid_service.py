import logging
import re
from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
import numpy as np
import torch

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ModuleNotFoundError:
    # speechbrain<1.0 exposes EncoderClassifier under speechbrain.pretrained
    from speechbrain.pretrained import EncoderClassifier

log = logging.getLogger("utils.lid_service")

_ALIAS_MAP = {
    "hindi": "hi",
    "hin": "hi",
    "hi-in": "hi",
    "telugu": "te",
    "tel": "te",
    "te-in": "te",
    "tamil": "ta",
    "tam": "ta",
    "ta-in": "ta",
    "marathi": "mr",
    "mar": "mr",
    "mr-in": "mr",
    "gujarati": "gu",
    "guj": "gu",
    "gu-in": "gu",
    "kannada": "kn",
    "kan": "kn",
    "kn-in": "kn",
    "malayalam": "ml",
    "mal": "ml",
    "ml-in": "ml",
    "punjabi": "pa",
    "pan": "pa",
    "pa-in": "pa",
    "bengali": "bn",
    "ben": "bn",
    "bn-in": "bn",
    "urdu": "ur",
    "urd": "ur",
    "ur-in": "ur",
    "odia": "or",
    "or-in": "or",
    "assamese": "as",
    "as-in": "as",
    "konkani": "kok",
    "kok-in": "kok",
    "sanskrit": "sa",
    "sa-in": "sa",
    "nepali": "ne",
    "nep": "ne",
    "ne-in": "ne",
    "sindhi": "sd",
    "snd": "sd",
    "sd-in": "sd",
}

@dataclass(frozen=True)
class LIDResult:
    language: Optional[str]
    confidence: float
    raw_label: str
    normalized_label: str


class SpeechBrainLIDService:
    def __init__(self, source: str = "speechbrain/lang-id-voxlingua107-ecapa", savedir: str = "models/lid_model"):
        self.device = "cpu"
        self.classifier: Optional[EncoderClassifier] = None
        self.source = source.strip()
        self.savedir = savedir.strip()
        self.last_error = ""

    def load_model(self) -> bool:
        if self.classifier is not None:
            return True
        log.info("Loading LID model from %s on CPU...", self.source)
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=self.source,
                savedir=self.savedir,
                run_opts={"device": self.device},
            )
            self.last_error = ""
            log.info("LID model loaded successfully on CPU")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            log.error("Failed to load LID model: %s", exc)
            self.classifier = None
            return False

    @staticmethod
    def normalize_language_label(label: str) -> str:
        raw = (label or "").strip().lower().replace("_", "-")
        normalized = re.sub(r"[^a-z-]+", "-", raw)
        normalized = re.sub(r"-+", "-", normalized).strip("-")
        return normalized

    @staticmethod
    def map_to_supported_code(raw_label: str, supported_languages: set[str]) -> Optional[str]:
        normalized = SpeechBrainLIDService.normalize_language_label(raw_label)
        base = normalized.split("-", 1)[0] if normalized else ""

        candidates: list[str] = []
        for candidate in (
            normalized,
            base,
            _ALIAS_MAP.get(normalized, ""),
            _ALIAS_MAP.get(base, ""),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if candidate in supported_languages:
                return candidate
        return None

    def identify_turn_language(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        supported_languages: set[str],
    ) -> LIDResult:
        if not self.classifier:
            raise RuntimeError("LID model is not loaded")
        if sample_rate != 16000:
            raise ValueError("LID supports only 16kHz audio")

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            raise ValueError("Empty audio bytes for LID")

        signal = torch.from_numpy(audio_np).to(self.device).unsqueeze(0)
        prediction = self.classifier.classify_batch(signal)
        
        # prediction[1] is the log-probability tensor
        score = prediction[1]
        confidence = float(torch.exp(score).item())
        
        raw_label = str(prediction[3][0])
        normalized_label = self.normalize_language_label(raw_label)
        language = self.map_to_supported_code(raw_label, supported_languages)

        return LIDResult(
            language=language,
            confidence=confidence,
            raw_label=raw_label,
            normalized_label=normalized_label,
        )

    def vote_file_language(
        self,
        turns: list[Any],
        channel_clips: dict[int, Path],
        denoised_path: Path,
        job_dir: Path,
        supported_languages: set[str],
        default_lang: str = "hi-IN",
    ) -> str:
        """Process each speech turn to vote, applying confidence threshold and majority voting."""
        if not self.classifier:
            log.warning("LID model is not loaded. Voting falls back to default language: %s", default_lang)
            return default_lang

        from .audio_processing import read_pcm16_wav, slice_wav
        from collections import Counter

        votes = []
        locale_map = {"hi": "hi-IN", "te": "te-IN", "ta": "ta-IN", "mr": "mr-IN"}

        for index, turn in enumerate(turns[:10]):
            clip_source = channel_clips.get(turn.channel, denoised_path) if turn.channel is not None else denoised_path
            temp_turn_wav = job_dir / f"temp_lid_turn_{index:04d}.wav"
            try:
                slice_wav(clip_source, temp_turn_wav, turn.start_sec, turn.end_sec)
                pcm, sr = read_pcm16_wav(temp_turn_wav)
                
                res = self.identify_turn_language(pcm, sr, supported_languages)
                
                # Option 3: Confidence Threshold Gating
                if res.confidence >= 0.70 and res.language:
                    voted_lang = res.language
                else:
                    voted_lang = "hi"
                
                votes.append(voted_lang)
            except Exception as e:
                log.warning("Failed to perform LID classification on turn %d: %s", index, e)
                votes.append("hi")
            finally:
                try:
                    if temp_turn_wav.exists():
                        temp_turn_wav.unlink()
                except Exception:
                    pass

        if votes:
            winning_base = Counter(votes).most_common(1)[0][0]
            winning_locale = locale_map.get(winning_base, f"{winning_base}-IN")
            log.info("LID voting round complete. Turn votes: %s. Winner: %s", dict(Counter(votes)), winning_locale)
            return winning_locale

        return default_lang
