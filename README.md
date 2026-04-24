# RunPod Voice Translation App

Push-to-talk voice translation app for RunPod with a browser mic UI, FastAPI backend, trilingual translation output, and optional text-to-speech playback.

The current repo is an app-first deployment, not a notebook-only snapshot. The live server boots from `app.py`, serves `templates/index.html`, and loads the translation plus TTS services on startup.

## Current Flow

1. Hold `Space` or press the microphone button in the browser.
2. The browser records PCM audio and uploads a WAV file to `POST /api/translate`.
3. Whisper produces:
   - the original-language transcript
   - an English translation
4. OpusMT translates the English text into:
   - Indonesian
   - Japanese
5. The UI renders all three text outputs:
   - source transcript
   - English
   - Indonesian
   - Japanese
6. The browser can request generated speech from `POST /api/tts`.

## Models In This Codebase

### Active default models

These are the models loaded by default from [`services/config.py`](/home/ravhi/runpod_jupyter/services/config.py:72) and used by the live app pipeline:

| Purpose | Default model ID | Where used |
| --- | --- | --- |
| Speech-to-text + English translation | `openai/whisper-large-v3` | [`services/whisper_service.py`](/home/ravhi/runpod_jupyter/services/whisper_service.py:43) |
| English -> Indonesian MT | `Helsinki-NLP/opus-mt-en-id` | [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:16) |
| English -> Japanese MT | `Helsinki-NLP/opus-mt-en-jap` | [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:16) |
| English TTS | `facebook/mms-tts-eng` | [`services/tts_service.py`](/home/ravhi/runpod_jupyter/services/tts_service.py:153) |
| Indonesian TTS | `facebook/mms-tts-ind` | [`services/tts_service.py`](/home/ravhi/runpod_jupyter/services/tts_service.py:153) |
| Japanese TTS | `suno/bark-small` | [`services/tts_service.py`](/home/ravhi/runpod_jupyter/services/tts_service.py:153) |

### Supported configurable model variants

The code also explicitly supports these model IDs or model swaps:

- Whisper aliases supported by the `faster-whisper` backend:
  - `openai/whisper-large-v3`
  - `openai/whisper-large-v3-turbo`
  - `openai/whisper-large-v2`
  - `openai/whisper-large`
  - `openai/whisper-medium`
  - `openai/whisper-small`
  - `openai/whisper-base`
  - `openai/whisper-tiny`
- Japanese Bark can be upgraded from `suno/bark-small` to `suno/bark` via `TTS_JA_MODEL_ID`.
- All active model IDs are environment-variable configurable.

## Backend Architecture

- [`app.py`](/home/ravhi/runpod_jupyter/app.py:1): FastAPI entrypoint, startup lifecycle, API routes
- [`services/pipeline.py`](/home/ravhi/runpod_jupyter/services/pipeline.py:1): orchestrates Whisper and OpusMT
- [`services/whisper_service.py`](/home/ravhi/runpod_jupyter/services/whisper_service.py:1): transcription and English translation
- [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:1): parallel EN -> ID and EN -> JA translation
- [`services/tts_service.py`](/home/ravhi/runpod_jupyter/services/tts_service.py:1): multilingual speech synthesis
- [`services/audio.py`](/home/ravhi/runpod_jupyter/services/audio.py:1): decode, chunking, preprocessing
- [`templates/index.html`](/home/ravhi/runpod_jupyter/templates/index.html:1): single-page browser UI
- [`start.sh`](/home/ravhi/runpod_jupyter/start.sh:1): simple RunPod launch command

## API Surface

### `POST /api/translate`

Multipart form fields:

- `audio`: WAV upload
- `utteranceId`: optional client-generated UUID

Returns:

- `utteranceId`
- `detectedLanguage`
- `detectedLanguageLabel`
- `transcript`
- `translationEnglish`
- `translationIndonesian`
- `translationJapanese`
- `audioDurationSeconds`
- `processingSeconds`

### `POST /api/tts`

Multipart form fields:

- `text`
- `language`

Returns `audio/wav` with headers:

- `X-Duration`
- `X-Language`

## Run On RunPod

1. Copy this folder to your pod.
2. Open HTTP port `8880` in RunPod networking or exposed ports.
3. Install a GPU-enabled PyTorch build that matches your pod image. Example for CUDA 12.4:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the app:

```bash
bash start.sh
```

Or directly:

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8880
```

6. Open the RunPod public HTTP URL for port `8880`.

## Environment Variables

### Server

- `HOST=0.0.0.0`
- `PORT=8880`

### Whisper

- `WHISPER_BACKEND=faster-whisper`
- `WHISPER_MODEL_ID=openai/whisper-large-v3`
- `WHISPER_COMPUTE_TYPE=auto`
- `WHISPER_NUM_BEAMS=1`
- `WHISPER_CPU_THREADS=4`
- `WHISPER_CONCURRENCY=2`
- `WHISPER_CHUNK_LENGTH_SECONDS=12`
- `WHISPER_SHORT_AUDIO_THRESHOLD_SECONDS=12.0`
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT=false`
- `WHISPER_VAD_FILTER=false`
- `WHISPER_VAD_MIN_SILENCE_MS=350`

### OpusMT

- `OPUS_ID_MODEL_ID=Helsinki-NLP/opus-mt-en-id`
- `OPUS_JA_MODEL_ID=Helsinki-NLP/opus-mt-en-jap`
- `OPUS_NUM_BEAMS=2`

### TTS

- `TTS_EN_MODEL_ID=facebook/mms-tts-eng`
- `TTS_ID_MODEL_ID=facebook/mms-tts-ind`
- `TTS_JA_MODEL_ID=suno/bark-small`
- `TTS_JA_VOICE=v2/ja_speaker_0`
- `TTS_SPEAKING_RATE=1.0`

### Shared runtime/audio

- `HF_TOKEN=...`
- `HUGGINGFACE_HUB_TOKEN=...`
- `MODEL_DEVICE=cuda`
- `MODEL_DTYPE=float16`
- `CHUNK_SECONDS=15`
- `CHUNK_OVERLAP_SECONDS=0.75`
- `TRIM_SILENCE=true`
- `SILENCE_THRESHOLD_RATIO=0.02`
- `SILENCE_PADDING_MS=180`
- `NORMALIZE_AUDIO=true`

If any selected model is gated or private, authenticate first with `hf auth login` or export `HF_TOKEN`.

## Frontend Notes

- The UI is a single-page app served from `templates/index.html`.
- It supports both keyboard push-to-talk and pointer/touch recording.
- The frontend renders trilingual text columns and can trigger TTS playback per language.
- The page title and UI branding currently say `Nexto StreamWave`.

## Notes

- The app startup logs the exact loaded model IDs for Whisper, OpusMT, and TTS.
- The current codebase does not include a checked-in notebook file in this directory.
- No runtime inference validation was executed as part of this README update.
