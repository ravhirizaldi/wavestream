from __future__ import annotations

import unittest
from types import SimpleNamespace

from services.common import language_label
from services.opus_service import OpusMTService
from services.tts_service import _LANG_ALIASES


class _InlineFuture:
    def __init__(self, value: str) -> None:
        self._value = value

    def result(self) -> str:
        return self._value


class _InlineExecutor:
    def submit(self, func, *args):
        return _InlineFuture(func(*args))


def _make_service() -> OpusMTService:
    service = OpusMTService(SimpleNamespace(mt_backend="opus", opus_pt_target_token=">>pt_BR<<"))
    service._executor = _InlineExecutor()
    service._id_tokenizer = "id"
    service._ja_tokenizer = "ja"
    service._pt_tokenizer = "pt"
    service._tl_tokenizer = "tl"
    service._id_en_tokenizer = "id_en"
    service._ja_en_tokenizer = "ja_en"
    service._pt_en_tokenizer = "pt_en"
    service._tl_en_tokenizer = "tl_en"
    return service


def _make_nllb_service() -> OpusMTService:
    return OpusMTService(SimpleNamespace(mt_backend="nllb"))


class LanguageSupportTests(unittest.TestCase):
    def test_language_labels_include_new_languages(self) -> None:
        self.assertEqual(language_label("pt-BR"), "Portuguese (Brazil)")
        self.assertEqual(language_label("tgl"), "Filipino")
        self.assertEqual(language_label("fil"), "Filipino")

    def test_tts_aliases_include_new_languages(self) -> None:
        self.assertEqual(_LANG_ALIASES["pt"], "pt")
        self.assertEqual(_LANG_ALIASES["pt_br"], "pt")
        self.assertEqual(_LANG_ALIASES["por"], "pt")
        self.assertEqual(_LANG_ALIASES["tl"], "tl")
        self.assertEqual(_LANG_ALIASES["tgl"], "tl")
        self.assertEqual(_LANG_ALIASES["fil"], "tl")

    def test_translate_reuses_portuguese_source_and_translates_other_targets(self) -> None:
        service = _make_service()
        calls: list[tuple[str, str | None]] = []

        def fake_translate(text, tokenizer, model, target_token=None):
            calls.append((tokenizer, target_token))
            return f"{tokenizer}:{target_token or ''}:{text}"

        service._translate_text = fake_translate  # type: ignore[method-assign]

        result = service.translate(
            english_text="hello",
            detected_language="pt-BR",
            source_portuguese="ola",
        )

        self.assertEqual(result.portuguese, "ola")
        self.assertEqual(result.indonesian, "id::hello")
        self.assertEqual(result.japanese, "ja::hello")
        self.assertEqual(result.filipino, "tl::hello")
        self.assertNotIn(("pt", ">>pt_BR<<"), calls)

    def test_translate_uses_portuguese_target_token_for_english_source(self) -> None:
        service = _make_service()
        calls: list[tuple[str, str | None]] = []

        def fake_translate(text, tokenizer, model, target_token=None):
            calls.append((tokenizer, target_token))
            return f"{tokenizer}:{target_token or ''}:{text}"

        service._translate_text = fake_translate  # type: ignore[method-assign]

        result = service.translate(english_text="hello", detected_language="en")

        self.assertEqual(result.portuguese, "pt:>>pt_BR<<:hello")
        self.assertIn(("pt", ">>pt_BR<<"), calls)

    def test_translate_to_english_uses_new_source_models(self) -> None:
        service = _make_service()

        def fake_translate(text, tokenizer, model, target_token=None):
            return f"{tokenizer}:{text}"

        service._translate_text = fake_translate  # type: ignore[method-assign]

        self.assertEqual(service.translate_to_english("ola", "por"), "pt_en:ola")
        self.assertEqual(service.translate_to_english("kumusta", "fil"), "tl_en:kumusta")

    def test_nllb_translate_routes_english_to_all_target_languages(self) -> None:
        service = _make_nllb_service()
        calls: list[tuple[str, str, str]] = []

        def fake_translate(text, source_language, target_language):
            calls.append((text, source_language, target_language))
            return f"{source_language}->{target_language}:{text}"

        service._translate_nllb = fake_translate  # type: ignore[method-assign]

        result = service.translate(english_text="hello", detected_language="en")

        self.assertEqual(result.indonesian, "en->id:hello")
        self.assertEqual(result.japanese, "en->ja:hello")
        self.assertEqual(result.portuguese, "en->pt:hello")
        self.assertEqual(result.filipino, "en->tl:hello")
        self.assertEqual(
            calls,
            [
                ("hello", "en", "id"),
                ("hello", "en", "ja"),
                ("hello", "en", "pt"),
                ("hello", "en", "tl"),
            ],
        )

    def test_nllb_translate_to_english_uses_detected_language(self) -> None:
        service = _make_nllb_service()

        def fake_translate(text, source_language, target_language):
            return f"{source_language}->{target_language}:{text}"

        service._translate_nllb = fake_translate  # type: ignore[method-assign]

        self.assertEqual(service.translate_to_english("kumusta", "fil"), "fil->en:kumusta")
        self.assertIsNone(service.translate_to_english("bonjour", "fr"))


if __name__ == "__main__":
    unittest.main()
