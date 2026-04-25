from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from services.config import Settings, load_settings
from services.pipeline import TranslationPipeline
from services.tts_service import TTSService

BASE_DIR = Path(__file__).resolve().parent
settings = load_settings()


class TranslationResponse(BaseModel):
    utteranceId: str
    detectedLanguage: str
    detectedLanguageLabel: str
    transcript: str
    translationEnglish: str
    translationIndonesian: str
    translationJapanese: str
    audioDurationSeconds: float
    processingSeconds: float


def _log_startup(current_settings: Settings, pipeline: TranslationPipeline) -> None:
    print("")
    print("=== RunPod Translation App ===")
    print(f"Listening on http://{current_settings.host}:{current_settings.port}")
    print(f"Whisper backend   : {pipeline.whisper.backend_name}")
    print(f"Whisper model     : {current_settings.whisper_model_id}")
    print(f"Whisper compute   : {pipeline.whisper.runtime_compute_type}")
    print(f"Whisper beams     : {current_settings.whisper_num_beams}")
    print(f"Whisper concur.   : {current_settings.whisper_concurrency}")
    print(f"OpusMT EN→ID     : {current_settings.opus_id_model_id}")
    print(f"OpusMT EN→JA     : {current_settings.opus_ja_model_id}")
    print(f"OpusMT ID→EN     : {current_settings.opus_id_en_model_id}")
    print(f"OpusMT JA→EN     : {current_settings.opus_ja_en_model_id}")
    print(f"Opus beams        : {current_settings.opus_num_beams}")
    print(f"TTS EN            : {current_settings.tts_en_model_id}")
    print(f"TTS JA            : {current_settings.tts_ja_model_id}  voice={current_settings.tts_ja_voice}")
    print(f"TTS ID            : {current_settings.tts_id_model_id}")
    print(f"Device            : {pipeline.whisper.device}")
    print(f"Dtype             : {pipeline.whisper.torch_dtype}")
    print("")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = TranslationPipeline(settings)
    tts      = TTSService(settings)
    await run_in_threadpool(pipeline.load)
    await run_in_threadpool(tts.load)
    app.state.pipeline = pipeline
    app.state.tts      = tts
    _log_startup(settings, pipeline)
    try:
        yield
    finally:
        pipeline.opus.shutdown()


app = FastAPI(title="RunPod PTT Translation App", lifespan=lifespan)


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/api/translate", response_model=TranslationResponse)
async def translate_audio(
    request: Request,
    audio: UploadFile = File(...),
    utterance_id: str | None = Form(default=None, alias="utteranceId"),
) -> TranslationResponse:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload was empty.")

    try:
        payload = await run_in_threadpool(
            request.app.state.pipeline.process_audio,
            audio_bytes,
            utterance_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc

    return TranslationResponse(
        utteranceId=payload.utterance_id,
        detectedLanguage=payload.detected_language,
        detectedLanguageLabel=payload.detected_language_label,
        transcript=payload.transcript,
        translationEnglish=payload.translation_english,
        translationIndonesian=payload.translation_indonesian,
        translationJapanese=payload.translation_japanese,
        audioDurationSeconds=payload.audio_duration_seconds,
        processingSeconds=payload.processing_seconds,
    )


@app.post("/api/tts")
async def text_to_speech(
    request: Request,
    text: str = Form(...),
    language: str = Form(default="en"),
) -> Response:
    """Synthesize speech for the given text and language, returns audio/wav."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        result = await run_in_threadpool(
            request.app.state.tts.synthesize,
            text.strip(),
            language,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    return Response(
        content=result.audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Duration": f"{result.duration_seconds:.3f}",
            "X-Language": result.language,
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", settings.host),
        port=int(os.getenv("PORT", str(settings.port))),
    )
