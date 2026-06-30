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


class _FakeTensor:
    def to(self, *_args, **_kwargs):
        return self


class _FakeTokenizer:
    unk_token_id = -1

    def __init__(self) -> None:
        self.src_lang = ""
        self.last_text = ""

    def set_src_lang_special_tokens(self, lang: str) -> None:
        self.src_lang = lang

    def __call__(self, text, **_kwargs):
        self.last_text = text
        return {"input_ids": _FakeTensor()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return 17 if token else self.unk_token_id

    def batch_decode(self, _tokens, skip_special_tokens=True):
        return [f"translated:{self.last_text}"]


class _FakeModel:
    def generate(self, **_kwargs):
        return [[1, 2, 3]]


def _make_service() -> OpusMTService:
    service = OpusMTService(
        SimpleNamespace(
            mt_backend="opus",
            opus_pt_target_token=">>pt_BR<<",
            opus_ms_target_token=">>zsm_Latn<<",
        )
    )
    service._executor = _InlineExecutor()
    service._id_tokenizer = "id"
    service._ja_tokenizer = "ja"
    service._pt_tokenizer = "pt"
    service._tl_tokenizer = "tl"
    service._ms_tokenizer = "ms"
    service._id_en_tokenizer = "id_en"
    service._ja_en_tokenizer = "ja_en"
    service._pt_en_tokenizer = "pt_en"
    service._tl_en_tokenizer = "tl_en"
    service._ms_en_tokenizer = "ms_en"
    return service


def _make_nllb_service() -> OpusMTService:
    return OpusMTService(
        SimpleNamespace(
            mt_backend="nllb",
            opus_num_beams=1,
            opus_max_new_tokens=64,
            opus_length_penalty=1.0,
        )
    )


class LanguageSupportTests(unittest.TestCase):
    def test_language_labels_include_new_languages(self) -> None:
        self.assertEqual(language_label("pt-BR"), "Portuguese (Brazil)")
        self.assertEqual(language_label("tgl"), "Filipino")
        self.assertEqual(language_label("fil"), "Filipino")
        self.assertEqual(language_label("ms"), "Malay (Malaysia)")
        self.assertEqual(language_label("zsm"), "Malay (Malaysia)")

    def test_tts_aliases_include_new_languages(self) -> None:
        self.assertEqual(_LANG_ALIASES["pt"], "pt")
        self.assertEqual(_LANG_ALIASES["pt_br"], "pt")
        self.assertEqual(_LANG_ALIASES["por"], "pt")
        self.assertEqual(_LANG_ALIASES["tl"], "tl")
        self.assertEqual(_LANG_ALIASES["tgl"], "tl")
        self.assertEqual(_LANG_ALIASES["fil"], "tl")
        self.assertEqual(_LANG_ALIASES["ms"], "ms")
        self.assertEqual(_LANG_ALIASES["msa"], "ms")
        self.assertEqual(_LANG_ALIASES["zlm"], "ms")

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
        self.assertEqual(result.malay, "ms:>>zsm_Latn<<:hello")
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

    def test_translate_reuses_malay_source_and_skips_malay_target(self) -> None:
        service = _make_service()
        calls: list[tuple[str, str | None]] = []

        def fake_translate(text, tokenizer, model, target_token=None):
            calls.append((tokenizer, target_token))
            return f"{tokenizer}:{target_token or ''}:{text}"

        service._translate_text = fake_translate  # type: ignore[method-assign]

        result = service.translate(
            english_text="hello",
            detected_language="zsm",
            source_malay="selamat datang",
        )

        self.assertEqual(result.malay, "selamat datang")
        self.assertNotIn(("ms", ">>zsm_Latn<<"), calls)

    def test_translate_uses_malay_target_token_for_english_source(self) -> None:
        service = _make_service()
        calls: list[tuple[str, str | None]] = []

        def fake_translate(text, tokenizer, model, target_token=None):
            calls.append((tokenizer, target_token))
            return f"{tokenizer}:{target_token or ''}:{text}"

        service._translate_text = fake_translate  # type: ignore[method-assign]

        result = service.translate(english_text="hello", detected_language="en")

        self.assertEqual(result.malay, "ms:>>zsm_Latn<<:hello")
        self.assertIn(("ms", ">>zsm_Latn<<"), calls)

    def test_translate_to_english_uses_new_source_models(self) -> None:
        service = _make_service()

        def fake_translate(text, tokenizer, model, target_token=None):
            return f"{tokenizer}:{text}"

        service._translate_text = fake_translate  # type: ignore[method-assign]

        self.assertEqual(service.translate_to_english("ola", "por"), "pt_en:ola")
        self.assertEqual(service.translate_to_english("kumusta", "fil"), "tl_en:kumusta")
        self.assertEqual(service.translate_to_english("selamat", "zsm"), "ms_en:selamat")

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
        self.assertEqual(result.malay, "en->ms:hello")
        self.assertEqual(
            calls,
            [
                ("hello", "en", "id"),
                ("hello", "en", "ja"),
                ("hello", "en", "pt"),
                ("hello", "en", "tl"),
                ("hello", "en", "ms"),
            ],
        )

    def test_nllb_translate_to_english_uses_detected_language(self) -> None:
        service = _make_nllb_service()

        def fake_translate(text, source_language, target_language):
            return f"{source_language}->{target_language}:{text}"

        service._translate_nllb = fake_translate  # type: ignore[method-assign]

        self.assertEqual(service.translate_to_english("kumusta", "fil"), "fil->en:kumusta")
        self.assertEqual(service.translate_to_english("selamat", "zsm"), "zsm->en:selamat")
        self.assertIsNone(service.translate_to_english("bonjour", "fr"))

    def test_nllb_splits_long_translation_text_into_smaller_units(self) -> None:
        service = _make_nllb_service()
        text = (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen. "
            "short sentence."
        )

        self.assertEqual(
            service._split_translation_units(text),
            [
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen",
                "nineteen.",
                "short sentence.",
            ],
        )

    def test_nllb_unit_translation_uses_local_text_argument(self) -> None:
        service = _make_nllb_service()
        service.device = "cpu"  # type: ignore[assignment]
        service._nllb_tokenizer = _FakeTokenizer()
        service._nllb_model = _FakeModel()

        self.assertEqual(
            service._translate_nllb_unit(" local text ", "en", "id"),
            "translated:local text",
        )


if __name__ == "__main__":
    unittest.main()
