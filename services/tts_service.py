from __future__ import annotations

import io
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoTokenizer, BarkModel, VitsModel

from services.common import detect_device
from services.config import Settings

# ─────────────────────────────────────────────────────────────────────────────
# Language code normalisation
# ─────────────────────────────────────────────────────────────────────────────
_LANG_ALIASES: dict[str, str] = {
    "en": "en", "eng": "en",
    "ja": "ja", "jpn": "ja",
    "id": "id", "ind": "id",
}
_DEFAULT_LANG = "en"

# Languages that use the Bark backend instead of VITS
_BARK_LANGS: frozenset[str] = frozenset({"ja"})


# ─────────────────────────────────────────────────────────────────────────────
# Abstract backend
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes      # WAV bytes (16-bit PCM)
    sample_rate: int
    duration_seconds: float
    language: str


class _Backend(ABC):
    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> tuple[np.ndarray, int]:
        """Return (audio_f32, sample_rate)."""


# ─────────────────────────────────────────────────────────────────────────────
# VITS backend  (MMS models — EN, ID)
# ─────────────────────────────────────────────────────────────────────────────

class _VitsBackend(_Backend):
    """
    Facebook MMS-TTS VITS model.  Fast (~200 ms/sentence on GPU), low VRAM.

    Available MMS models on HuggingFace:
        facebook/mms-tts-eng    English   (~130 MB)
        facebook/mms-tts-ind    Indonesian (~130 MB)
    """

    def __init__(self, model_id: str, hf_token: str | None, device: torch.device) -> None:
        print(f"[TTS/VITS] Loading: {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        try:
            mdl = VitsModel.from_pretrained(model_id, use_safetensors=True, token=hf_token)
        except (OSError, ValueError):
            mdl = VitsModel.from_pretrained(model_id, token=hf_token)
        mdl = mdl.to(device)
        mdl.eval()
        self.tokenizer = tok
        self.model     = mdl
        self.device    = device

    def synthesize(self, text: str, **_) -> tuple[np.ndarray, int]:  # type: ignore[override]
        inputs = self.tokenizer(text.strip(), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
        audio = out.waveform.squeeze().cpu().numpy().astype(np.float32)
        return audio, self.model.config.sampling_rate


# ─────────────────────────────────────────────────────────────────────────────
# Bark backend  (Suno Bark — Japanese and any other multilingual need)
# ─────────────────────────────────────────────────────────────────────────────

class _BarkBackend(_Backend):
    """
    Suno Bark TTS — multilingual autoregressive model.

    Japanese voice presets:  v2/ja_speaker_0 … v2/ja_speaker_9
    Model options:
        suno/bark-small   ~1.5 GB  (faster, good quality)   ← default
        suno/bark         ~6 GB    (slower, better quality)

    Output: 24 000 Hz mono float32.
    """

    SAMPLE_RATE = 24_000

    def __init__(
        self,
        model_id: str,
        hf_token: str | None,
        device: torch.device,
        default_voice: str = "v2/ja_speaker_0",
    ) -> None:
        print(f"[TTS/Bark] Loading: {model_id}  (voice={default_voice})")
        self.processor     = AutoProcessor.from_pretrained(model_id, token=hf_token)

        _load_kwargs = dict(
            token       = hf_token,
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32,
        )
        # Prefer safetensors — avoids torch.load / CVE-2025-32434 on torch < 2.6
        try:
            self.model = BarkModel.from_pretrained(
                model_id, use_safetensors=True, **_load_kwargs
            )
        except (OSError, ValueError):
            self.model = BarkModel.from_pretrained(model_id, **_load_kwargs)

        self.model         = self.model.to(device)
        self.model.eval()
        self.device        = device
        self.default_voice = default_voice

    def synthesize(self, text: str, voice: str | None = None, **_) -> tuple[np.ndarray, int]:  # type: ignore[override]
        import warnings

        voice   = voice or self.default_voice
        inputs  = self.processor(text, voice_preset=voice)
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}

        # Bark's multi-stage generation triggers several known harmless warnings
        # from the transformers library. Suppress them to keep logs clean.
        _bark_warning_patterns = [
            "A custom logits processor",
            "Passing `generation_config` together",
            "The attention mask and the pad token id",
            "The attention mask is not set",
            "Both `max_new_tokens`",
        ]
        with torch.inference_mode():
            with warnings.catch_warnings():
                for pattern in _bark_warning_patterns:
                    warnings.filterwarnings("ignore", message=f".*{pattern}.*")
                audio_tensor = self.model.generate(**inputs)

        audio = audio_tensor.cpu().numpy().squeeze().astype(np.float32)
        # Bark output can be in int16 range; normalise to [-1, 1] if needed
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak
        return audio, self.SAMPLE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Public service
# ─────────────────────────────────────────────────────────────────────────────

class TTSService:
    """
    Multi-backend TTS service.

    Backend assignment:
        English     → VITS (facebook/mms-tts-eng)        fast, ~130 MB
        Indonesian  → VITS (facebook/mms-tts-ind)         fast, ~130 MB
        Japanese    → Bark (suno/bark-small by default)   good quality, ~1.5 GB

    All model IDs and Bark voice preset are configurable via env vars:
        TTS_EN_MODEL_ID   TTS_ID_MODEL_ID   TTS_JA_MODEL_ID
        TTS_JA_VOICE      TTS_SPEAKING_RATE
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device:   torch.device | None = None
        self._backends: dict[str, _Backend] = {}
        self._lock     = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        self.device, _ = detect_device(
            self.settings.preferred_device,
            self.settings.preferred_dtype,
        )

        # ── VITS models ────────────────────────────────────
        for lang, model_id in (
            ("en", self.settings.tts_en_model_id),
            ("id", self.settings.tts_id_model_id),
        ):
            self._backends[lang] = _VitsBackend(model_id, self.settings.hf_token, self.device)

        # ── Bark model for Japanese ────────────────────────
        self._backends["ja"] = _BarkBackend(
            model_id      = self.settings.tts_ja_model_id,
            hf_token      = self.settings.hf_token,
            device        = self.device,
            default_voice = self.settings.tts_ja_voice,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def synthesize(self, text: str, language: str) -> TTSResult:
        lang    = _LANG_ALIASES.get(language.lower().strip(), _DEFAULT_LANG)
        backend = self._backends.get(lang) or self._backends[_DEFAULT_LANG]

        if not text or not text.strip():
            silence = np.zeros(int(16_000 * 0.15), dtype=np.float32)
            return _encode(silence, 16_000, lang)

        with self._lock:
            if isinstance(backend, _BarkBackend):
                audio, sr = backend.synthesize(text, voice=self.settings.tts_ja_voice)
            else:
                audio, sr = backend.synthesize(text)

        # Optional speed change via resampling
        rate = float(self.settings.tts_speaking_rate)
        if abs(rate - 1.0) > 0.05:
            audio = _resample_speed(audio, rate)

        return _encode(audio, sr, lang)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample_speed(audio: np.ndarray, rate: float) -> np.ndarray:
    orig_len = len(audio)
    new_len  = max(1, int(orig_len / rate))
    return np.interp(
        np.linspace(0, orig_len - 1, new_len),
        np.arange(orig_len),
        audio,
    ).astype(np.float32)


def _encode(audio: np.ndarray, sr: int, lang: str) -> TTSResult:
    audio = np.clip(audio, -1.0, 1.0)
    buf   = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return TTSResult(
        audio_bytes      = buf.getvalue(),
        sample_rate      = sr,
        duration_seconds = len(audio) / sr,
        language         = lang,
    )
