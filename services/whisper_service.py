from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from services.audio import chunk_audio, load_audio, preprocess_audio
from services.common import (
    clean_output_text,
    detect_device,
    detect_language_with_fallback,
    dominant_language,
    infer_language_from_text,
    language_label,
    merge_transcript_segments,
    normalize_lang_key,
)
from services.config import Settings

_FASTER_WHISPER_MODEL_ALIASES = {
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "turbo",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large": "large",
    "openai/whisper-medium": "medium",
    "openai/whisper-small": "small",
    "openai/whisper-base": "base",
    "openai/whisper-tiny": "tiny",
}


@dataclass(frozen=True)
class WhisperTranscriptionResult:
    detected_language: str
    detected_language_label: str
    transcript: str
    translation_english: str
    audio_duration_seconds: float
    processed_duration_seconds: float
    chunk_count: int
    stage_timings: dict[str, float]


class WhisperService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend_name = settings.whisper_backend.strip().lower()
        self.runtime_compute_type = "uninitialized"
        self.device: torch.device | None = None
        self.torch_dtype: torch.dtype | None = None
        self.processor: WhisperProcessor | None = None
        self.model: WhisperForConditionalGeneration | None = None
        self.faster_model = None
        self.sample_rate = 16000
        self._semaphore = threading.BoundedSemaphore(
            max(1, self.settings.whisper_concurrency)
        )

    def load(self) -> None:
        self.device, self.torch_dtype = detect_device(
            self.settings.preferred_device, self.settings.preferred_dtype
        )
        if self.backend_name == "faster-whisper":
            self._load_faster_whisper()
            return
        if self.backend_name != "transformers":
            raise ValueError(
                f"Unsupported whisper backend '{self.settings.whisper_backend}'. "
                "Use 'faster-whisper' or 'transformers'."
            )

        model_kwargs: dict[str, object] = {
            "low_cpu_mem_usage": True,
            "token": self.settings.hf_token,
        }
        if self.device.type != "cpu":
            model_kwargs["dtype"] = self.torch_dtype

        self.processor = WhisperProcessor.from_pretrained(
            self.settings.whisper_model_id,
            token=self.settings.hf_token,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.settings.whisper_model_id,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = int(
            getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        )
        self.runtime_compute_type = str(self.torch_dtype).replace("torch.", "")

    def transcribe_bytes(self, audio_bytes: bytes) -> WhisperTranscriptionResult:
        if self.backend_name == "faster-whisper":
            return self._transcribe_bytes_faster_whisper(audio_bytes)
        return self._transcribe_bytes_transformers(audio_bytes)

    def _transcribe_bytes_transformers(self, audio_bytes: bytes) -> WhisperTranscriptionResult:
        if (
            self.processor is None
            or self.model is None
            or self.device is None
            or self.torch_dtype is None
        ):
            raise RuntimeError("Whisper service is not loaded.")

        stage_timings: dict[str, float] = {}
        decode_started_at = time.perf_counter()
        audio, input_sample_rate = load_audio(audio_bytes)
        stage_timings["audio_decode"] = time.perf_counter() - decode_started_at

        preprocess_started_at = time.perf_counter()
        processed_audio, raw_duration_seconds, processed_duration_seconds = (
            preprocess_audio(
                audio=audio,
                input_sample_rate=input_sample_rate,
                target_sample_rate=self.sample_rate,
                settings=self.settings,
            )
        )
        stage_timings["preprocess"] = time.perf_counter() - preprocess_started_at
        if processed_audio.size == 0 or processed_duration_seconds <= 0:
            raise ValueError("The recording did not contain usable audio.")

        chunks = chunk_audio(
            audio=processed_audio,
            sample_rate=self.sample_rate,
            chunk_seconds=self.settings.chunk_seconds,
            overlap_seconds=self.settings.chunk_overlap_seconds,
        )

        transcript_chunks: list[str] = []
        english_chunks: list[str] = []
        detected_languages: list[str] = []

        transcribe_config = self._generation_config(task="transcribe")
        translate_config = self._generation_config(task="translate")
        whisper_transcribe_seconds = 0.0
        whisper_translate_seconds = 0.0

        for chunk in chunks:
            if chunk.size < int(self.sample_rate * 0.25):
                continue

            model_inputs = self.processor(
                audio=chunk,
                sampling_rate=self.sample_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )

            # Move input_features to device once
            input_features = model_inputs["input_features"].to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=self.device)

            with torch.inference_mode():
                # ── Encoder runs ONCE per chunk ──────────────────────────────
                transcribe_started_at = time.perf_counter()
                encoder_outputs = self.model.model.encoder(
                    input_features=input_features,
                    attention_mask=attention_mask,
                )

                # ── Decoder pass 1: original language transcription ──────────
                gen_transcribe = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    generation_config=transcribe_config,
                )
                whisper_transcribe_seconds += time.perf_counter() - transcribe_started_at

                # ── Decoder pass 2: built-in translation → English ───────────
                translate_started_at = time.perf_counter()
                gen_translate = self.model.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    generation_config=translate_config,
                )
                whisper_translate_seconds += time.perf_counter() - translate_started_at

            # Language detection from the decoded string (with special tokens).
            # Newer transformers may not include forced prefix tokens in the
            # generated token IDs, so we decode the raw output and regex-match
            # Whisper's <|language|> tag instead of scanning token IDs.
            raw_with_special = self.processor.batch_decode(
                gen_transcribe, skip_special_tokens=False
            )[0]
            transcript_text = clean_output_text(
                self.processor.batch_decode(gen_transcribe, skip_special_tokens=True)[0]
            )
            detected_languages.append(
                detect_language_with_fallback(raw_with_special, transcript_text)
            )
            english_text = clean_output_text(
                self.processor.batch_decode(gen_translate, skip_special_tokens=True)[0]
            )

            if transcript_text:
                transcript_chunks.append(transcript_text)
            if english_text:
                english_chunks.append(english_text)

        transcript = merge_transcript_segments(transcript_chunks)
        translation_english = merge_transcript_segments(english_chunks)

        if not transcript:
            raise ValueError("No speech was detected after preprocessing the recording.")

        detected_language = dominant_language(detected_languages)
        stage_timings["whisper_transcribe"] = whisper_transcribe_seconds
        stage_timings["whisper_translate"] = whisper_translate_seconds
        return WhisperTranscriptionResult(
            detected_language=detected_language,
            detected_language_label=language_label(detected_language),
            transcript=transcript,
            translation_english=translation_english or transcript,
            audio_duration_seconds=raw_duration_seconds,
            processed_duration_seconds=processed_duration_seconds,
            chunk_count=len(chunks),
            stage_timings=stage_timings,
        )

    def _transcribe_bytes_faster_whisper(self, audio_bytes: bytes) -> WhisperTranscriptionResult:
        if self.faster_model is None:
            raise RuntimeError("Whisper service is not loaded.")

        stage_timings: dict[str, float] = {}
        decode_started_at = time.perf_counter()
        audio, input_sample_rate = load_audio(audio_bytes)
        stage_timings["audio_decode"] = time.perf_counter() - decode_started_at

        preprocess_started_at = time.perf_counter()
        processed_audio, raw_duration_seconds, processed_duration_seconds = (
            preprocess_audio(
                audio=audio,
                input_sample_rate=input_sample_rate,
                target_sample_rate=self.sample_rate,
                settings=self.settings,
            )
        )
        stage_timings["preprocess"] = time.perf_counter() - preprocess_started_at
        if processed_audio.size == 0 or processed_duration_seconds <= 0:
            raise ValueError("The recording did not contain usable audio.")

        transcribe_segments, transcribe_info, transcribe_seconds = (
            self._run_faster_whisper_task(
                audio=processed_audio,
                task="transcribe",
                language=None,
                processed_duration_seconds=processed_duration_seconds,
            )
        )
        transcript = merge_transcript_segments(
            [clean_output_text(segment.text) for segment in transcribe_segments]
        )
        if not transcript:
            raise ValueError("No speech was detected after preprocessing the recording.")

        detected_language = normalize_lang_key(getattr(transcribe_info, "language", None))
        if detected_language == "und":
            detected_language = infer_language_from_text(transcript)

        translation_seconds = 0.0
        if detected_language in {"en", "eng"}:
            translation_english = transcript
        else:
            english_segments, _, translation_seconds = self._run_faster_whisper_task(
                audio=processed_audio,
                task="translate",
                language=detected_language if detected_language != "und" else None,
                processed_duration_seconds=processed_duration_seconds,
            )
            translation_english = merge_transcript_segments(
                [clean_output_text(segment.text) for segment in english_segments]
            ) or transcript

        stage_timings["whisper_transcribe"] = transcribe_seconds
        stage_timings["whisper_translate"] = translation_seconds
        return WhisperTranscriptionResult(
            detected_language=detected_language,
            detected_language_label=language_label(detected_language),
            transcript=transcript,
            translation_english=translation_english,
            audio_duration_seconds=raw_duration_seconds,
            processed_duration_seconds=processed_duration_seconds,
            chunk_count=max(1, len(transcribe_segments)),
            stage_timings=stage_timings,
        )

    def _load_faster_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'faster-whisper' backend is selected but the package is not installed. "
                "Install dependencies from requirements.txt before starting the app."
            ) from exc

        self.runtime_compute_type = self._resolve_faster_whisper_compute_type()
        faster_device = self._resolve_faster_whisper_device()
        resolved_model_id = _FASTER_WHISPER_MODEL_ALIASES.get(
            self.settings.whisper_model_id,
            self.settings.whisper_model_id,
        )
        self.faster_model = WhisperModel(
            resolved_model_id,
            device=faster_device,
            compute_type=self.runtime_compute_type,
            cpu_threads=max(1, self.settings.whisper_cpu_threads),
        )
        self.sample_rate = 16000

    def _run_faster_whisper_task(
        self,
        *,
        audio,
        task: str,
        language: str | None,
        processed_duration_seconds: float,
    ):
        if self.faster_model is None:
            raise RuntimeError("Whisper service is not loaded.")

        is_short_audio = (
            processed_duration_seconds
            <= self.settings.whisper_short_audio_threshold_seconds
        )
        kwargs: dict[str, object] = {
            "task": task,
            "beam_size": max(1, self.settings.whisper_num_beams),
            "best_of": 1,
            "condition_on_previous_text": (
                False
                if is_short_audio
                else self.settings.whisper_condition_on_previous_text
            ),
            "without_timestamps": True,
            "vad_filter": self.settings.whisper_vad_filter and not is_short_audio,
        }
        if not is_short_audio:
            kwargs["chunk_length"] = max(
                1, int(self.settings.whisper_chunk_length_seconds)
            )
        if kwargs["vad_filter"]:
            kwargs["vad_parameters"] = {
                "min_silence_duration_ms": self.settings.whisper_vad_min_silence_ms,
            }
        if language:
            kwargs["language"] = language

        started_at = time.perf_counter()
        with self._semaphore:
            segments, info = self.faster_model.transcribe(audio, **kwargs)
            segment_list = list(segments)
        return segment_list, info, time.perf_counter() - started_at

    def _resolve_faster_whisper_device(self) -> str:
        if self.device is None:
            raise RuntimeError("Whisper device has not been initialized.")
        if self.device.type == "cuda":
            return "cuda"
        return "cpu"

    def _resolve_faster_whisper_compute_type(self) -> str:
        configured = self.settings.whisper_compute_type.strip().lower()
        if configured and configured != "auto":
            return configured
        if self.device and self.device.type == "cuda":
            return "float16"
        return "int8"

    def _generation_config(self, task: str = "transcribe"):
        """Return a generation config for the given task (transcribe or translate)."""
        if self.model is None:
            raise RuntimeError("Whisper service is not loaded.")
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = False
        generation_config.num_beams = self.settings.whisper_num_beams
        generation_config.task = task
        for attr in ("temperature", "top_k", "top_p"):
            if hasattr(generation_config, attr):
                setattr(generation_config, attr, None)
        return generation_config
