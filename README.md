# RunPod Voice Translation App

Push-to-talk voice translation app for RunPod with a browser mic UI, FastAPI backend, trilingual translation output, and optional text-to-speech playback.

The current repo is an app-first deployment, not a notebook-only snapshot. The live server boots from `app.py`, serves `templates/index.html`, and loads the translation plus TTS services on startup.

## Current Flow

1. Hold `Space` or press the microphone button in the browser.
2. The browser records PCM audio and uploads a WAV file to `POST /api/translate`.
3. Whisper produces the original-language transcript.
4. OpusMT produces English from Japanese or Indonesian text, avoiding a second Whisper audio pass.
5. OpusMT translates the English text into:
   - Indonesian
   - Japanese
6. The UI renders all three text outputs:
   - source transcript
   - English
   - Indonesian
   - Japanese
7. The browser can request generated speech from `POST /api/tts`.

## Models In This Codebase

### Active default models

These are the models loaded by default from [`services/config.py`](/home/ravhi/runpod_jupyter/services/config.py:72) and used by the live app pipeline:

| Purpose | Default model ID | Where used |
| --- | --- | --- |
| Speech-to-text | `openai/whisper-large-v3` | [`services/whisper_service.py`](/home/ravhi/runpod_jupyter/services/whisper_service.py:43) |
| English -> Indonesian MT | `Helsinki-NLP/opus-mt-en-id` | [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:16) |
| English -> Japanese MT | `Helsinki-NLP/opus-mt-en-jap` | [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:16) |
| Indonesian -> English MT | `Helsinki-NLP/opus-mt-id-en` | [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:16) |
| Japanese -> English MT | `staka/fugumt-ja-en` | [`services/opus_service.py`](/home/ravhi/runpod_jupyter/services/opus_service.py:16) |
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
- `WHISPER_NUM_BEAMS=5`
- `WHISPER_CPU_THREADS=4`
- `WHISPER_CONCURRENCY=2`
- `WHISPER_CHUNK_LENGTH_SECONDS=12`
- `WHISPER_SHORT_AUDIO_THRESHOLD_SECONDS=12.0`
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT=false`
- `WHISPER_VAD_FILTER=true`
- `WHISPER_VAD_MIN_SILENCE_MS=350`
- `WHISPER_COMPRESSION_RATIO_THRESHOLD=2.2`
- `WHISPER_LOG_PROB_THRESHOLD=-0.8`
- `WHISPER_NO_SPEECH_THRESHOLD=0.6`

### OpusMT

- `OPUS_ID_MODEL_ID=Helsinki-NLP/opus-mt-en-id`
- `OPUS_JA_MODEL_ID=Helsinki-NLP/opus-mt-en-jap`
- `OPUS_ID_EN_MODEL_ID=Helsinki-NLP/opus-mt-id-en`
- `OPUS_JA_EN_MODEL_ID=staka/fugumt-ja-en` (FuGuMT, drop-in MarianMT replacement — dramatically more faithful than `Helsinki-NLP/opus-mt-ja-en` on conversational JA)
- `OPUS_NUM_BEAMS=2` (FuGuMT-ja-en is most faithful at low beams; Helsinki-NLP models in the other directions are insensitive to 2 vs 5)
- `OPUS_MAX_NEW_TOKENS=384`
- `OPUS_NO_REPEAT_NGRAM_SIZE=3`
- `OPUS_LENGTH_PENALTY=1.0`

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
- `SILENCE_THRESHOLD_RATIO=0.01`
- `SILENCE_THRESHOLD_FLOOR=0.0015`
- `SILENCE_PADDING_MS=180`
- `NORMALIZE_AUDIO=true`

If any selected model is gated or private, authenticate first with `hf auth login` or export `HF_TOKEN`.

## Hardware Specs Requirements

These are practical deployment requirements for the current default stack in this repo:

- `openai/whisper-large-v3`
- `Helsinki-NLP/opus-mt-en-id`
- `Helsinki-NLP/opus-mt-en-jap`
- `Helsinki-NLP/opus-mt-id-en`
- `staka/fugumt-ja-en`
- `facebook/mms-tts-eng`
- `facebook/mms-tts-ind`
- `suno/bark-small`

They are not hard-coded startup checks, but they reflect what you should provision if you want the app to feel responsive.

### Minimum usable target

- GPU: NVIDIA CUDA GPU with at least `16 GB` VRAM
- CPU: `8 vCPU` or better
- System RAM: `16 GB`
- Disk: `25 GB` free SSD space for environment, model cache, and temporary files
- Network: stable internet for initial Hugging Face model download/authentication

This tier is the practical floor for the shipped defaults. It may still feel tight during model load or if you keep `WHISPER_CONCURRENCY=2`.

### Recommended production target

- GPU: NVIDIA CUDA GPU with `24 GB` VRAM or more
- CPU: `8-16 vCPU`
- System RAM: `24-32 GB`
- Disk: `40 GB+` free SSD space

This is the safer target for smoother startup, fewer memory-pressure issues, and better interactive latency with the default Whisper, four OpusMT models, and multilingual TTS loaded together.

### Low-spec or fallback mode

- CPU-only or Apple `mps` execution is technically possible because the runtime falls back from CUDA when needed, but it should be treated as development/debug mode rather than the intended interactive deployment path.
- GPUs below `16 GB` VRAM are likely to require reducing model weight or runtime pressure, for example:
  - lower `WHISPER_CONCURRENCY`
  - switch to a smaller Whisper model
  - disable or replace Bark-based Japanese TTS

### Provisioning notes

- The Japanese TTS path is the heaviest extra TTS load in the default setup because it uses `suno/bark-small`.
- The four OpusMT models and two MMS TTS models are comparatively lighter, but they still add memory and download size on top of Whisper.
- Use SSD-backed storage. First boot can be much slower if the model cache has to be downloaded into a cold environment.
- Match your PyTorch install to the CUDA version on the machine or RunPod image before starting the service.

## Notes

- The app startup logs the exact loaded model IDs for Whisper, OpusMT, and TTS.
- The current codebase does not include a checked-in notebook file in this directory.
- No runtime inference validation was executed as part of this README update.
