from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from transformers import MarianMTModel, MarianTokenizer

from services.common import clean_output_text, detect_device
from services.config import Settings


@dataclass(frozen=True)
class OpusTranslationResult:
    indonesian: str
    japanese: str


class OpusMTService:
    """
    Two purpose-built Helsinki-NLP MarianMT models (~300 MB each):
      - opus-mt-en-id  : English → Indonesian
      - opus-mt-en-jap : English → Japanese

    Both translations run in parallel via a thread pool, so total latency
    is max(t_id, t_ja) instead of t_id + t_ja.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device: torch.device | None = None

        # English → Indonesian
        self._id_tokenizer: MarianTokenizer | None = None
        self._id_model: MarianMTModel | None = None

        # English → Japanese
        self._ja_tokenizer: MarianTokenizer | None = None
        self._ja_model: MarianMTModel | None = None

        # Indonesian → English
        self._id_en_tokenizer: MarianTokenizer | None = None
        self._id_en_model: MarianMTModel | None = None

        # Japanese → English
        self._ja_en_tokenizer: MarianTokenizer | None = None
        self._ja_en_model: MarianMTModel | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        self.device, _ = detect_device(
            self.settings.preferred_device, self.settings.preferred_dtype
        )
        self._id_tokenizer, self._id_model = self._load_model(
            self.settings.opus_id_model_id
        )
        self._ja_tokenizer, self._ja_model = self._load_model(
            self.settings.opus_ja_model_id
        )
        self._id_en_tokenizer, self._id_en_model = self._load_model(
            self.settings.opus_id_en_model_id
        )
        self._ja_en_tokenizer, self._ja_en_model = self._load_model(
            self.settings.opus_ja_en_model_id
        )

    def _load_model(
        self, model_id: str
    ) -> tuple[MarianTokenizer, MarianMTModel]:
        tokenizer = MarianTokenizer.from_pretrained(
            model_id, token=self.settings.hf_token
        )

        load_kwargs: dict = {
            "token": self.settings.hf_token,
        }

        # Prefer safetensors — avoids torch.load entirely (CVE-2025-32434).
        # Fall back to the standard path if the model hub entry has no
        # safetensors shard (older Helsinki-NLP models only ship .bin files).
        try:
            model = MarianMTModel.from_pretrained(
                model_id, use_safetensors=True, **load_kwargs
            )
        except (OSError, ValueError):
            # .safetensors not available for this model — requires torch >= 2.6
            model = MarianMTModel.from_pretrained(
                model_id, **load_kwargs
            )

        model_dtype = torch.float16 if self.device and self.device.type == "cuda" else torch.float32
        model = model.to(dtype=model_dtype, device=self.device)
        model.eval()
        return tokenizer, model

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def translate(
        self,
        english_text: str,
        detected_language: str,
        source_indonesian: str = "",
        source_japanese: str = "",
    ) -> OpusTranslationResult:
        """
        Translate English text into Indonesian and Japanese in parallel.

        If the source language IS one of the targets we skip that OPUS call
        and reuse the original transcript directly (faster + more accurate).

        Args:
            english_text:       Whisper's built-in EN translation.
            detected_language:  ISO code from Whisper (e.g. "id", "ja", "en").
            source_indonesian:  Original transcript to reuse when source=ID.
            source_japanese:    Original transcript to reuse when source=JA.
        """
        lang = (detected_language or "").lower()

        need_id = lang not in ("id", "ind")
        need_ja = lang not in ("ja", "jpn")

        indonesian: str = source_indonesian if not need_id else ""
        japanese: str = source_japanese if not need_ja else ""

        # Nothing to translate — both targets are the source language
        if not need_id and not need_ja:
            return OpusTranslationResult(indonesian=indonesian, japanese=japanese)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: dict = {}
            if need_id:
                futures["id"] = pool.submit(
                    self._translate_text,
                    english_text,
                    self._id_tokenizer,
                    self._id_model,
                )
            if need_ja:
                futures["ja"] = pool.submit(
                    self._translate_text,
                    english_text,
                    self._ja_tokenizer,
                    self._ja_model,
                )

            if "id" in futures:
                indonesian = futures["id"].result()
            if "ja" in futures:
                japanese = futures["ja"].result()

        return OpusTranslationResult(indonesian=indonesian, japanese=japanese)

    def translate_to_english(self, text: str, detected_language: str) -> str | None:
        lang = (detected_language or "").lower()
        if lang in ("en", "eng"):
            return text
        if lang in ("id", "ind"):
            return self._translate_text(text, self._id_en_tokenizer, self._id_en_model)
        if lang in ("ja", "jpn"):
            return self._translate_text(text, self._ja_en_tokenizer, self._ja_en_model)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _translate_text(
        self,
        text: str,
        tokenizer: MarianTokenizer | None,
        model: MarianMTModel | None,
    ) -> str:
        if not text or not text.strip():
            return ""
        if tokenizer is None or model is None or self.device is None:
            raise RuntimeError("OpusMT model not loaded.")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                num_beams=self.settings.opus_num_beams,
                max_length=512,
            )

        raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return clean_output_text(raw)
