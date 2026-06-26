from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer

from services.common import (
    clean_output_text,
    clear_max_length_default,
    detect_device,
    normalize_lang_key,
    silence_transformers_max_length_warning,
)
from services.config import Settings

_PUNCTUATION_ONLY_PATTERN = re.compile(r"^[\W_]+$", flags=re.UNICODE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")
_MAX_NLLB_UNIT_WORDS = 18
_ENGLISH_CODES = frozenset({"en", "eng"})
_INDONESIAN_CODES = frozenset({"id", "ind"})
_JAPANESE_CODES = frozenset({"ja", "jpn"})
_PORTUGUESE_CODES = frozenset({"pt", "pt_br", "por"})
_FILIPINO_CODES = frozenset({"tl", "tgl", "fil"})

_NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "id": "ind_Latn",
    "ind": "ind_Latn",
    "ja": "jpn_Jpan",
    "jpn": "jpn_Jpan",
    "pt": "por_Latn",
    "pt_br": "por_Latn",
    "por": "por_Latn",
    "tl": "tgl_Latn",
    "tgl": "tgl_Latn",
    "fil": "tgl_Latn",
}


@dataclass(frozen=True)
class OpusTranslationResult:
    indonesian: str
    japanese: str
    portuguese: str
    filipino: str


class OpusMTService:
    """
    Purpose-built Helsinki-NLP MarianMT models for app target languages.

    Target translations run in parallel via a thread pool, so total latency
    is bounded by the slowest target instead of the sum of every target.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device: torch.device | None = None
        self.mt_backend = str(getattr(settings, "mt_backend", "opus")).strip().lower()

        # NLLB multilingual translation backend
        self._nllb_tokenizer: Any | None = None
        self._nllb_model: Any | None = None
        self._nllb_lock = Lock()

        # English → Indonesian
        self._id_tokenizer: MarianTokenizer | None = None
        self._id_model: MarianMTModel | None = None

        # English → Japanese
        self._ja_tokenizer: MarianTokenizer | None = None
        self._ja_model: MarianMTModel | None = None

        # English → Portuguese
        self._pt_tokenizer: MarianTokenizer | None = None
        self._pt_model: MarianMTModel | None = None

        # English → Filipino / Tagalog
        self._tl_tokenizer: MarianTokenizer | None = None
        self._tl_model: MarianMTModel | None = None

        # Indonesian → English
        self._id_en_tokenizer: MarianTokenizer | None = None
        self._id_en_model: MarianMTModel | None = None

        # Japanese → English
        self._ja_en_tokenizer: MarianTokenizer | None = None
        self._ja_en_model: MarianMTModel | None = None

        # Portuguese → English
        self._pt_en_tokenizer: MarianTokenizer | None = None
        self._pt_en_model: MarianMTModel | None = None

        # Filipino / Tagalog → English
        self._tl_en_tokenizer: MarianTokenizer | None = None
        self._tl_en_model: MarianMTModel | None = None

        # Persistent worker pool — avoids spinning up two threads per request.
        # max_workers=4 because at most we run four EN→target translations.
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="opus-mt"
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        silence_transformers_max_length_warning()
        self.device, _ = detect_device(
            self.settings.preferred_device, self.settings.preferred_dtype
        )
        if self._use_nllb:
            self._nllb_tokenizer, self._nllb_model = self._load_nllb_model(
                self.settings.nllb_model_id
            )
            return

        self._id_tokenizer, self._id_model = self._load_model(
            self.settings.opus_id_model_id
        )
        self._ja_tokenizer, self._ja_model = self._load_model(
            self.settings.opus_ja_model_id
        )
        self._pt_tokenizer, self._pt_model = self._load_model(
            self.settings.opus_pt_model_id
        )
        self._tl_tokenizer, self._tl_model = self._load_model(
            self.settings.opus_tl_model_id
        )
        self._id_en_tokenizer, self._id_en_model = self._load_model(
            self.settings.opus_id_en_model_id
        )
        self._ja_en_tokenizer, self._ja_en_model = self._load_model(
            self.settings.opus_ja_en_model_id
        )
        self._pt_en_tokenizer, self._pt_en_model = self._load_model(
            self.settings.opus_pt_en_model_id
        )
        self._tl_en_tokenizer, self._tl_en_model = self._load_model(
            self.settings.opus_tl_en_model_id
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
        # Marian's default generation_config carries max_length=512. Clearing
        # it here makes the explicit max_new_tokens we pass at call time the
        # sole length signal, suppressing the noisy
        # `Both max_new_tokens and max_length seem to have been set` log line.
        clear_max_length_default(getattr(model, "generation_config", None))
        return tokenizer, model

    def _load_nllb_model(self, model_id: str) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=self.settings.hf_token
        )

        load_kwargs: dict = {
            "token": self.settings.hf_token,
        }

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, use_safetensors=True, **load_kwargs
            )
        except (OSError, ValueError):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, **load_kwargs
            )

        model_dtype = torch.float16 if self.device and self.device.type == "cuda" else torch.float32
        model = model.to(dtype=model_dtype, device=self.device)
        model.eval()
        clear_max_length_default(getattr(model, "generation_config", None))
        return tokenizer, model

    @property
    def _use_nllb(self) -> bool:
        return self.mt_backend == "nllb"

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def translate(
        self,
        english_text: str,
        detected_language: str,
        source_indonesian: str = "",
        source_japanese: str = "",
        source_portuguese: str = "",
        source_filipino: str = "",
    ) -> OpusTranslationResult:
        """
        Translate English text into every non-source target language.

        If the source language IS one of the targets we skip that OPUS call
        and reuse the original transcript directly (faster + more accurate).

        Args:
            english_text:       Whisper's built-in EN translation.
            detected_language:  ISO code from Whisper (e.g. "id", "ja", "en").
            source_indonesian:  Original transcript to reuse when source=ID.
            source_japanese:    Original transcript to reuse when source=JA.
            source_portuguese:  Original transcript to reuse when source=PT.
            source_filipino:    Original transcript to reuse when source=TL.
        """
        lang = normalize_lang_key(detected_language)

        if self._use_nllb:
            return self._translate_nllb_targets(
                english_text=english_text,
                detected_language=lang,
                source_indonesian=source_indonesian,
                source_japanese=source_japanese,
                source_portuguese=source_portuguese,
                source_filipino=source_filipino,
            )

        need_id = lang not in _INDONESIAN_CODES
        need_ja = lang not in _JAPANESE_CODES
        need_pt = lang not in _PORTUGUESE_CODES
        need_tl = lang not in _FILIPINO_CODES

        indonesian: str = source_indonesian if not need_id else ""
        japanese: str = source_japanese if not need_ja else ""
        portuguese: str = source_portuguese if not need_pt else ""
        filipino: str = source_filipino if not need_tl else ""

        # Defensive no-op if every target is somehow already satisfied.
        if not any((need_id, need_ja, need_pt, need_tl)):
            return OpusTranslationResult(
                indonesian=indonesian,
                japanese=japanese,
                portuguese=portuguese,
                filipino=filipino,
            )

        futures: dict = {}
        if need_id:
            futures["id"] = self._executor.submit(
                self._translate_text,
                english_text,
                self._id_tokenizer,
                self._id_model,
            )
        if need_ja:
            futures["ja"] = self._executor.submit(
                self._translate_text,
                english_text,
                self._ja_tokenizer,
                self._ja_model,
            )
        if need_pt:
            futures["pt"] = self._executor.submit(
                self._translate_text,
                english_text,
                self._pt_tokenizer,
                self._pt_model,
                self.settings.opus_pt_target_token,
            )
        if need_tl:
            futures["tl"] = self._executor.submit(
                self._translate_text,
                english_text,
                self._tl_tokenizer,
                self._tl_model,
            )

        if "id" in futures:
            indonesian = futures["id"].result()
        if "ja" in futures:
            japanese = futures["ja"].result()
        if "pt" in futures:
            portuguese = futures["pt"].result()
        if "tl" in futures:
            filipino = futures["tl"].result()

        return OpusTranslationResult(
            indonesian=indonesian,
            japanese=japanese,
            portuguese=portuguese,
            filipino=filipino,
        )

    def translate_to_english(self, text: str, detected_language: str) -> str | None:
        lang = normalize_lang_key(detected_language)
        if self._use_nllb:
            if lang not in _NLLB_LANG_CODES:
                return None
            return self._translate_nllb(text, lang, "en")

        if lang in _ENGLISH_CODES:
            return text
        if lang in _INDONESIAN_CODES:
            return self._translate_text(text, self._id_en_tokenizer, self._id_en_model)
        if lang in _JAPANESE_CODES:
            return self._translate_text(text, self._ja_en_tokenizer, self._ja_en_model)
        if lang in _PORTUGUESE_CODES:
            return self._translate_text(text, self._pt_en_tokenizer, self._pt_en_model)
        if lang in _FILIPINO_CODES:
            return self._translate_text(text, self._tl_en_tokenizer, self._tl_en_model)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _translate_nllb_targets(
        self,
        english_text: str,
        detected_language: str,
        source_indonesian: str = "",
        source_japanese: str = "",
        source_portuguese: str = "",
        source_filipino: str = "",
    ) -> OpusTranslationResult:
        lang = normalize_lang_key(detected_language)

        need_id = lang not in _INDONESIAN_CODES
        need_ja = lang not in _JAPANESE_CODES
        need_pt = lang not in _PORTUGUESE_CODES
        need_tl = lang not in _FILIPINO_CODES

        indonesian = source_indonesian if not need_id else ""
        japanese = source_japanese if not need_ja else ""
        portuguese = source_portuguese if not need_pt else ""
        filipino = source_filipino if not need_tl else ""

        if not any((need_id, need_ja, need_pt, need_tl)):
            return OpusTranslationResult(
                indonesian=indonesian,
                japanese=japanese,
                portuguese=portuguese,
                filipino=filipino,
            )

        if need_id:
            indonesian = self._translate_nllb(english_text, "en", "id")
        if need_ja:
            japanese = self._translate_nllb(english_text, "en", "ja")
        if need_pt:
            portuguese = self._translate_nllb(english_text, "en", "pt")
        if need_tl:
            filipino = self._translate_nllb(english_text, "en", "tl")

        return OpusTranslationResult(
            indonesian=indonesian,
            japanese=japanese,
            portuguese=portuguese,
            filipino=filipino,
        )

    def _translate_nllb(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        if not cleaned or _PUNCTUATION_ONLY_PATTERN.match(cleaned):
            return ""
        units = self._split_translation_units(cleaned)
        if len(units) > 1:
            translated_units = [
                self._translate_nllb_unit(unit, source_language, target_language)
                for unit in units
            ]
            return clean_output_text(" ".join(unit for unit in translated_units if unit))

        return self._translate_nllb_unit(cleaned, source_language, target_language)

    def _split_translation_units(self, text: str) -> list[str]:
        units = [
            part.strip()
            for part in _SENTENCE_SPLIT_PATTERN.split(text.strip())
            if part.strip()
        ]
        if not units:
            return []

        split_units: list[str] = []
        for unit in units:
            words = unit.split()
            if len(words) <= _MAX_NLLB_UNIT_WORDS:
                split_units.append(unit)
                continue
            for idx in range(0, len(words), _MAX_NLLB_UNIT_WORDS):
                split_units.append(" ".join(words[idx:idx + _MAX_NLLB_UNIT_WORDS]))
        return split_units

    def _translate_nllb_unit(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if self._nllb_tokenizer is None or self._nllb_model is None or self.device is None:
            raise RuntimeError("NLLB model not loaded.")

        source_code = _NLLB_LANG_CODES.get(normalize_lang_key(source_language))
        target_code = _NLLB_LANG_CODES.get(normalize_lang_key(target_language))
        if not source_code or not target_code:
            raise RuntimeError(
                f"NLLB language pair not configured: {source_language!r}->{target_language!r}"
            )

        with self._nllb_lock:
            tokenizer = self._nllb_tokenizer
            model = self._nllb_model

            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = source_code
            set_src_lang: Callable[[str], None] | None = getattr(
                tokenizer, "set_src_lang_special_tokens", None
            )
            if set_src_lang:
                set_src_lang(source_code)

            inputs = tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)
            if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
                raise RuntimeError(f"NLLB target language token not found: {target_code}")

            num_beams = max(1, self.settings.opus_num_beams)
            generate_kwargs: dict[str, object] = {
                "forced_bos_token_id": forced_bos_token_id,
                "num_beams": num_beams,
                "max_new_tokens": max(16, self.settings.opus_max_new_tokens),
                "length_penalty": self.settings.opus_length_penalty,
                "early_stopping": num_beams > 1,
            }

            with torch.inference_mode():
                tokens = model.generate(**inputs, **generate_kwargs)

            raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        return clean_output_text(raw)

    def _translate_text(
        self,
        text: str,
        tokenizer: MarianTokenizer | None,
        model: MarianMTModel | None,
        target_token: str | None = None,
    ) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        if not cleaned or _PUNCTUATION_ONLY_PATTERN.match(cleaned):
            # Marian readily hallucinates a translation for punctuation-only
            # input (e.g. "..." → "Tidak."). Bail early.
            return ""
        if tokenizer is None or model is None or self.device is None:
            raise RuntimeError("OpusMT model not loaded.")

        if target_token:
            cleaned = f"{target_token.strip()} {cleaned}"

        # padding=True on a single string is a no-op; skip it to avoid
        # tokenizer overhead per request.
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        num_beams = max(1, self.settings.opus_num_beams)
        generate_kwargs: dict[str, object] = {
            "num_beams": num_beams,
            "max_new_tokens": max(16, self.settings.opus_max_new_tokens),
            "length_penalty": self.settings.opus_length_penalty,
            # Stop the moment all beams hit EOS — major win when output is
            # much shorter than max_new_tokens.
            "early_stopping": num_beams > 1,
        }
        no_repeat = self.settings.opus_no_repeat_ngram_size
        if no_repeat and no_repeat > 0:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat

        with torch.inference_mode():
            tokens = model.generate(**inputs, **generate_kwargs)

        raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return clean_output_text(raw)
