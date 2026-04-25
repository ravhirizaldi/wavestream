from __future__ import annotations

import json
import re
from collections import Counter

import torch

TAG_ONLY_PATTERN = re.compile(r"^\s*\[[^\[\]]+\]\s*$")
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
WHISPER_LANGUAGE_TOKEN_PATTERN = re.compile(r"^<\|([a-z]{2,3})\|>$")
JAPANESE_SCRIPT_PATTERN = re.compile(r"[\u3040-\u30ff]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

LANGUAGE_LABELS = {
    "ar":  "Arabic",
    "cmn": "Mandarin",
    "de":  "German",
    "en":  "English",
    "eng": "English",
    "es":  "Spanish",
    "fr":  "French",
    "hi":  "Hindi",
    "id":  "Indonesian",
    "ind": "Indonesian",
    "it":  "Italian",
    "ja":  "Japanese",
    "jpn": "Japanese",
    "ko":  "Korean",
    "pt":  "Portuguese",
    "ru":  "Russian",
    "th":  "Thai",
    "uk":  "Ukrainian",
    "vi":  "Vietnamese",
    "zh":  "Chinese",
}

WHISPER_LANGUAGE_CODE_ALIASES = {
    "en":  "en",
    "es":  "es",
    "fr":  "fr",
    "de":  "de",
    "it":  "it",
    "pt":  "pt",
    "ru":  "ru",
    "ja":  "ja",
    "jpn": "ja",
    "ko":  "ko",
    "zh":  "zh",
    "cmn": "zh",
    "id":  "id",
    "ind": "id",
    "th":  "th",
    "vi":  "vi",
    "ar":  "ar",
    "hi":  "hi",
    "uk":  "uk",
}

ENGLISH_HINT_WORDS = frozenset({
    "a", "about", "and", "are", "can", "for", "from", "good", "hello",
    "how", "i", "i'm", "is", "it", "listen", "listening", "of", "please",
    "the", "this", "to", "we", "what", "you", "your",
})

INDONESIAN_HINT_WORDS = frozenset({
    "ada", "adalah", "apa", "baik", "bisa", "dan", "dari", "dengan", "di",
    "hal", "halo", "ini", "itu", "jepang", "kami", "kamu", "saya", "selamat",
    "tentang", "terima", "untuk", "yang",
})


def detect_device(preferred_device: str | None = None, preferred_dtype: str | None = None) -> tuple[torch.device, torch.dtype]:
    if preferred_device:
        device = torch.device(preferred_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if preferred_dtype:
        normalized = preferred_dtype.strip().lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in dtype_map:
            raise ValueError(f"Unsupported dtype '{preferred_dtype}'. Use float16, bfloat16, or float32.")
        return device, dtype_map[normalized]

    if device.type == "cuda":
        return device, torch.float16
    return device, torch.float32


def clean_output_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lang_key(language: str | None) -> str:
    if not language:
        return "und"
    return language.strip().lower().replace("-", "_")


def language_label(language_code: str | None) -> str:
    key = normalize_lang_key(language_code)
    return LANGUAGE_LABELS.get(key, key.upper() if key != "und" else "Unknown")


def normalized_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


_OVERLAP_NORMALIZE_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)


def _normalize_for_overlap(word: str) -> str:
    return _OVERLAP_NORMALIZE_PATTERN.sub("", word).lower()


def merge_transcript_segments(segments: list[str]) -> str:
    merged: list[str] = []
    for segment in segments:
        cleaned = clean_output_text(segment)
        if not cleaned:
            continue
        if not merged:
            merged.append(cleaned)
            continue
        previous_words = merged[-1].split()
        current_words = cleaned.split()
        overlap_size = 0
        max_overlap = min(16, len(previous_words), len(current_words))
        # Punctuation/casing-insensitive comparison so chunks like
        # "...hello," and "hello world" still detect the overlap.
        previous_norm = [_normalize_for_overlap(word) for word in previous_words]
        current_norm = [_normalize_for_overlap(word) for word in current_words]
        for candidate in range(max_overlap, 0, -1):
            left = previous_norm[-candidate:]
            right = current_norm[:candidate]
            # Skip overlaps composed entirely of dropped tokens (pure
            # punctuation chunks) — they would always match and falsely
            # eat real content.
            if any(left) and left == right:
                overlap_size = candidate
                break
        if overlap_size:
            cleaned = " ".join(current_words[overlap_size:])
        if cleaned:
            merged.append(cleaned)
    return " ".join(merged).strip()


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    minutes, seconds = divmod(total_seconds, 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def flatten_generated_tokens(generated: torch.Tensor) -> list[int]:
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    if isinstance(generated, tuple):
        generated = generated[0]
    if generated.ndim == 2:
        generated = generated[0]
    elif generated.ndim != 1:
        raise ValueError(f"Unexpected token shape: {tuple(generated.shape)}")
    return generated.detach().cpu().tolist()


def detect_language_from_decoded_text(decoded_with_special: str) -> str:
    """
    Extract the Whisper language code from the raw decoded string that
    still contains special tokens, e.g.:
        '<|startoftranscript|><|ja|><|transcribe|><|notimestamps|>text...'

    This is more reliable than scanning generated token IDs because
    newer versions of transformers may not include forced prefix tokens
    in the generate() output sequence.
    """
    _SKIP = frozenset({
        "transcribe", "translate", "notimestamps",
        "startoflm", "startoftranscript", "nospeech", "prev",
    })
    for match in re.finditer(r'<\|([a-z]{2,3})\|>', decoded_with_special):
        code = match.group(1).lower()
        if code not in _SKIP:
            return WHISPER_LANGUAGE_CODE_ALIASES.get(code, code)
    return "und"


def infer_language_from_text(text: str | None) -> str:
    cleaned = clean_output_text(text)
    if not cleaned:
        return "und"

    # Hiragana / Katakana make Japanese straightforward to identify.
    if JAPANESE_SCRIPT_PATTERN.search(cleaned):
        return "ja"

    # Kanji-only text is ambiguous between Japanese and Chinese, so leave it
    # unknown instead of forcing a bad label.
    if CJK_PATTERN.search(cleaned):
        return "und"

    words = normalized_words(cleaned)
    if not words:
        return "und"

    english_score = sum(word in ENGLISH_HINT_WORDS for word in words)
    indonesian_score = sum(word in INDONESIAN_HINT_WORDS for word in words)
    if english_score >= 2 and english_score > indonesian_score:
        return "en"
    if indonesian_score >= 2 and indonesian_score > english_score:
        return "id"
    return "und"


def detect_language_with_fallback(
    decoded_with_special: str,
    transcript_text: str | None = None,
) -> str:
    detected = detect_language_from_decoded_text(decoded_with_special)
    if detected != "und":
        return detected
    return infer_language_from_text(transcript_text)


# Keep old name as a thin alias for backwards compatibility
def detect_language_from_whisper_tokens(tokenizer, token_ids: list[int]) -> str:
    """Deprecated: prefer detect_language_from_decoded_text."""
    tokens = tokenizer.convert_ids_to_tokens(token_ids[:16]) or []
    fake_decoded = "".join(t or "" for t in tokens)
    return detect_language_from_decoded_text(fake_decoded)


def dominant_language(language_codes: list[str]) -> str:
    normalized = [normalize_lang_key(code) for code in language_codes if normalize_lang_key(code) != "und"]
    if not normalized:
        return "und"
    return Counter(normalized).most_common(1)[0][0]


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> dict[str, object]:
    stripped = strip_code_fences(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output did not contain a JSON object.")
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Model output JSON could not be parsed.") from exc


def validate_translation_payload(payload: dict[str, object]) -> tuple[str, str]:
    english = clean_output_text(str(payload.get("english", "") or ""))
    indonesian = clean_output_text(str(payload.get("indonesian", "") or ""))
    if not english or not indonesian:
        raise ValueError("Model output JSON must contain non-empty 'english' and 'indonesian' fields.")
    return english, indonesian
