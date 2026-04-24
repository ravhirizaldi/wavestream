from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from services.config import Settings


def load_audio(source: bytes | bytearray | memoryview) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(bytes(source)), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def trim_silence(audio: np.ndarray, sample_rate: int, threshold_ratio: float = 0.02, padding_ms: int = 180) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio
    threshold = max(peak * threshold_ratio, 0.003)
    active_indices = np.flatnonzero(np.abs(audio) >= threshold)
    if active_indices.size == 0:
        return audio
    pad = int(sample_rate * (padding_ms / 1000.0))
    start = max(0, int(active_indices[0]) - pad)
    end = min(audio.size, int(active_indices[-1]) + pad + 1)
    return audio[start:end]


def normalize_audio(audio: np.ndarray, peak_target: float = 0.92) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio
    scale = min(peak_target / peak, 8.0)
    return np.clip(audio * scale, -1.0, 1.0).astype(np.float32, copy=False)


def resample_audio(audio: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if original_sample_rate == target_sample_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration_seconds = audio.size / float(original_sample_rate)
    target_length = max(1, int(round(duration_seconds * target_sample_rate)))
    source_positions = np.linspace(0.0, audio.size - 1, num=audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, num=target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


def chunk_audio(audio: np.ndarray, sample_rate: int, chunk_seconds: float, overlap_seconds: float) -> list[np.ndarray]:
    if chunk_seconds <= 0:
        return [audio]
    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    if chunk_samples <= 0 or audio.size <= chunk_samples:
        return [audio]
    step = max(1, chunk_samples - overlap_samples)
    chunks: list[np.ndarray] = []
    start = 0
    while start < audio.size:
        end = min(audio.size, start + chunk_samples)
        chunks.append(audio[start:end])
        if end >= audio.size:
            break
        start += step
    return chunks


def preprocess_audio(
    audio: np.ndarray,
    input_sample_rate: int,
    target_sample_rate: int,
    settings: Settings,
) -> tuple[np.ndarray, float, float]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    raw_duration_seconds = audio.size / float(input_sample_rate)
    if input_sample_rate != target_sample_rate:
        audio = resample_audio(audio, input_sample_rate, target_sample_rate)
    if settings.trim_silence:
        audio = trim_silence(audio, target_sample_rate, settings.silence_threshold_ratio, settings.silence_padding_ms)
    if settings.normalize_audio:
        audio = normalize_audio(audio)
    processed_duration_seconds = audio.size / float(target_sample_rate) if target_sample_rate else 0.0
    return audio, raw_duration_seconds, processed_duration_seconds
