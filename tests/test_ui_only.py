from __future__ import annotations

import os
from io import BytesIO
import unittest

os.environ["UI_ONLY"] = "true"

import app as app_module
from fastapi import HTTPException, Request, UploadFile


class UiOnlyModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.lifespan = app_module.app.router.lifespan_context(app_module.app)
        await self.lifespan.__aenter__()
        self.request = Request({"type": "http", "app": app_module.app})

    async def asyncTearDown(self) -> None:
        await self.lifespan.__aexit__(None, None, None)

    async def test_ui_only_imports_no_ml_services_and_serves_index(self) -> None:
        self.assertFalse(hasattr(app_module, "TranslationPipeline"))
        self.assertFalse(hasattr(app_module, "TTSService"))
        self.assertIsNone(app_module.app.state.pipeline)
        self.assertIsNone(app_module.app.state.tts)

        response = await app_module.index()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(str(response.path).endswith("templates/index.html"))

    async def test_translate_returns_503_without_models(self) -> None:
        upload = UploadFile(filename="sample.wav", file=BytesIO(b"sample"))

        with self.assertRaises(HTTPException) as raised:
            await app_module.translate_audio(self.request, upload)

        self.assertEqual(raised.exception.status_code, 503)

    async def test_tts_returns_503_without_models(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await app_module.text_to_speech(self.request, "hello", "en")

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
