from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from services.config import Settings
from services.opus_service import OpusMTService
from services.whisper_service import WhisperService

# Languages that have native OpusMT targets; anything else falls back to EN output.
_JAPANESE_CODES  = frozenset({"ja", "jpn"})
_INDONESIAN_CODES = frozenset({"id", "ind"})


@dataclass(frozen=True)
class TranslationResponsePayload:
    utterance_id: str
    detected_language: str
    detected_language_label: str
    transcript: str
    translation_english: str
    translation_indonesian: str
    translation_japanese: str
    audio_duration_seconds: float
    processing_seconds: float
    stage_timings: dict[str, float] = field(default_factory=dict)


class TranslationPipeline:
    """
    Trilingual pipeline — Japanese / English / Indonesian.

    Flow
    ────
    1.  Whisper backend:
        · task=transcribe  →  original-language transcript
        · task=translate   →  English translation when the source is non-English

    2.  OpusMT (parallel, both ~50–150 ms each on GPU):
        · Helsinki-NLP/opus-mt-en-id   English → Indonesian
        · Helsinki-NLP/opus-mt-en-jap  English → Japanese

        Smart skip: if the source IS Indonesian or Japanese the model call is
        skipped and the original transcript is reused directly, saving time and
        preserving character-perfect output.

    Detected language  →  what gets reused vs translated
    ──────────────────────────────────────────────────────
    Japanese   (ja)   transcript=JA  EN=Whisper  ID=OPUS  JA=transcript
    Indonesian (id)   transcript=ID  EN=Whisper  ID=transcript  JA=OPUS
    English    (en)   transcript=EN  EN=Whisper  ID=OPUS  JA=OPUS
    Other             transcript=XX  EN=Whisper  ID=OPUS  JA=OPUS
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.whisper = WhisperService(settings)
        self.opus = OpusMTService(settings)

    def load(self) -> None:
        self.whisper.load()
        self.opus.load()

    def process_audio(
        self, audio_bytes: bytes, utterance_id: str | None = None
    ) -> TranslationResponsePayload:
        started_at = time.perf_counter()
        # ── Step 1: Whisper ──────────────────────────────────────────────
        transcription = self.whisper.transcribe_bytes(audio_bytes)
        lang = transcription.detected_language.lower()

        # ── Step 2: Determine what English text to feed OPUS ────────────
        english_text = transcription.translation_english
        stage_timings = dict(transcription.stage_timings)
        if not english_text:
            to_english_started_at = time.perf_counter()
            english_text = (
                self.opus.translate_to_english(
                    text=transcription.transcript,
                    detected_language=lang,
                )
                or ""
            )
            stage_timings["opus_to_english"] = time.perf_counter() - to_english_started_at

        if not english_text:
            fallback_started_at = time.perf_counter()
            english_text = self.whisper.translate_bytes_to_english(
                audio_bytes=audio_bytes,
                language=lang,
            )
            stage_timings["whisper_translate_fallback"] = (
                time.perf_counter() - fallback_started_at
            )

        # When source is JA or ID the original transcript IS that language.
        source_as_indonesian = transcription.transcript if lang in _INDONESIAN_CODES else ""
        source_as_japanese   = transcription.transcript if lang in _JAPANESE_CODES   else ""

        # ── Step 3: OpusMT — EN→ID and EN→JA in parallel ────────────────
        opus_started_at = time.perf_counter()
        opus_result = self.opus.translate(
            english_text=english_text,
            detected_language=lang,
            source_indonesian=source_as_indonesian,
            source_japanese=source_as_japanese,
        )
        stage_timings["opus"] = time.perf_counter() - opus_started_at
        total_processing_seconds = time.perf_counter() - started_at
        stage_timings["total"] = total_processing_seconds
        self._log_request_metrics(
            utterance_id=utterance_id,
            lang=lang,
            transcription=transcription,
            stage_timings=stage_timings,
        )

        return TranslationResponsePayload(
            utterance_id=utterance_id or str(uuid.uuid4()),
            detected_language=transcription.detected_language,
            detected_language_label=transcription.detected_language_label,
            transcript=transcription.transcript,
            translation_english=english_text,
            translation_indonesian=opus_result.indonesian,
            translation_japanese=opus_result.japanese,
            audio_duration_seconds=transcription.audio_duration_seconds,
            processing_seconds=total_processing_seconds,
            stage_timings=stage_timings,
        )

    def _log_request_metrics(
        self,
        utterance_id: str | None,
        lang: str,
        transcription,
        stage_timings: dict[str, float],
    ) -> None:
        timings = ", ".join(
            f"{name}={value:.3f}s"
            for name, value in stage_timings.items()
        )
        print(
            "[translate] "
            f"utterance_id={utterance_id or 'generated'} "
            f"lang={lang or 'und'} "
            f"audio={transcription.audio_duration_seconds:.2f}s "
            f"chunks={transcription.chunk_count} "
            f"{timings}"
        )
