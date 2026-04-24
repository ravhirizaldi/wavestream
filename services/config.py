from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    # ── Whisper (transcription + built-in EN translation) ──────────────────
    whisper_backend: str
    whisper_model_id: str
    whisper_compute_type: str
    whisper_num_beams: int
    whisper_cpu_threads: int
    whisper_concurrency: int
    whisper_chunk_length_seconds: int
    whisper_short_audio_threshold_seconds: float
    whisper_condition_on_previous_text: bool
    whisper_vad_filter: bool
    whisper_vad_min_silence_ms: int
    # ── OpusMT (EN → target language) ────────────────────────────────────────
    opus_id_model_id: str   # English → Indonesian
    opus_ja_model_id: str   # English → Japanese
    opus_num_beams: int
    # ── TTS (Text-to-Speech) ──────────────────────────────────────────────────
    tts_en_model_id: str     # VITS MMS model for English
    tts_ja_model_id: str     # Bark model for Japanese (multilingual)
    tts_ja_voice: str        # Bark voice preset, e.g. v2/ja_speaker_0
    tts_id_model_id: str     # VITS MMS model for Indonesian
    tts_speaking_rate: float # 1.0 = normal, 0.9 = slower, 1.1 = faster
    # ── Shared ────────────────────────────────────────────────────────────
    hf_token: str | None
    preferred_device: str | None
    preferred_dtype: str | None
    chunk_seconds: float
    chunk_overlap_seconds: float
    trim_silence: bool
    silence_threshold_ratio: float
    silence_padding_ms: int
    normalize_audio: bool


def load_settings() -> Settings:
    return Settings(
        host=_env_str("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8880),
        # Whisper
        whisper_backend=_env_str("WHISPER_BACKEND", "faster-whisper"),
        whisper_model_id=_env_str("WHISPER_MODEL_ID", "openai/whisper-large-v3"),
        whisper_compute_type=_env_str("WHISPER_COMPUTE_TYPE", "auto"),
        whisper_num_beams=_env_int("WHISPER_NUM_BEAMS", 1),
        whisper_cpu_threads=_env_int("WHISPER_CPU_THREADS", 4),
        whisper_concurrency=_env_int("WHISPER_CONCURRENCY", 2),
        whisper_chunk_length_seconds=_env_int("WHISPER_CHUNK_LENGTH_SECONDS", 12),
        whisper_short_audio_threshold_seconds=_env_float("WHISPER_SHORT_AUDIO_THRESHOLD_SECONDS", 12.0),
        whisper_condition_on_previous_text=_env_bool("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False),
        whisper_vad_filter=_env_bool("WHISPER_VAD_FILTER", False),
        whisper_vad_min_silence_ms=_env_int("WHISPER_VAD_MIN_SILENCE_MS", 350),
        # OpusMT — Helsinki-NLP MarianMT, purpose-built NMT, ~300 MB each
        opus_id_model_id=_env_str("OPUS_ID_MODEL_ID", "Helsinki-NLP/opus-mt-en-id"),
        opus_ja_model_id=_env_str("OPUS_JA_MODEL_ID", "Helsinki-NLP/opus-mt-en-jap"),
        opus_num_beams=_env_int("OPUS_NUM_BEAMS", 2),
        # TTS
        # EN + ID  : Facebook MMS-TTS VITS (~130 MB each)  — fast VITS synthesis
        # JA       : Suno Bark small (~1.5 GB)             — multilingual autoregressive
        tts_en_model_id=_env_str("TTS_EN_MODEL_ID", "facebook/mms-tts-eng"),
        tts_ja_model_id=_env_str("TTS_JA_MODEL_ID", "suno/bark-small"),
        tts_ja_voice=_env_str("TTS_JA_VOICE",       "v2/ja_speaker_0"),
        tts_id_model_id=_env_str("TTS_ID_MODEL_ID", "facebook/mms-tts-ind"),
        tts_speaking_rate=_env_float("TTS_SPEAKING_RATE", 1.0),
        # Shared
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        preferred_device=os.getenv("MODEL_DEVICE"),
        preferred_dtype=os.getenv("MODEL_DTYPE"),
        chunk_seconds=_env_float("CHUNK_SECONDS", 15.0),
        chunk_overlap_seconds=_env_float("CHUNK_OVERLAP_SECONDS", 0.75),
        trim_silence=_env_bool("TRIM_SILENCE", True),
        silence_threshold_ratio=_env_float("SILENCE_THRESHOLD_RATIO", 0.02),
        silence_padding_ms=_env_int("SILENCE_PADDING_MS", 180),
        normalize_audio=_env_bool("NORMALIZE_AUDIO", True),
    )
