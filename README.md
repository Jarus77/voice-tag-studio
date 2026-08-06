# voice-tag-studio

Localhost workbench for preparing Hindi/Hinglish TTS training data with inline
emotion tags. Paste a YouTube link → audio-only download → diarize → denoise →
segment → transcribe (srota) → run tag detectors one at a time → listen to
every intermediate + see tag placements on the transcript.

## Models

| Step | Model | Notes |
|---|---|---|
| Diarization | `pyannote/speaker-diarization-community-1` | **gated** — accept terms on its HF page once |
| Denoise | demucs `htdemucs` vocals stem | listening/QA only; fallback ffmpeg `afftdn` |
| ASR | `Surajgameramp/srota` (qwen3-asr-0.6b Hinglish FT) | runs in `.venv-asr` (transformers pin) |
| Word timings | torchaudio `MMS_FA` + uroman | + `pyannote/segmentation-3.0` VAD |
| [laughs] | PANNs CNN14 SED (AudioSet) | ~1.1 GB checkpoint auto-downloads |

## Policy (carried over from the voice project)

- The bestaudio **master download is archival** — never modified; 16 kHz mono
  PCM_16 is the working format cut from it.
- **Denoised audio never enters the dataset** and detectors read the ORIGINAL
  by default (htdemucs strips laughter — exactly what we tag).
- Precision over recall: candidates are cheap, wrong tags are poison. The
  audition ear is the gate.
- Tagged events stay EMBEDDED in flowing speech: segment reruns treat detected
  events ±0.5 s as no-cut zones.

## Setup

```bash
# main env (conda base python): fastapi yt-dlp demucs panns_inference noisereduce
pip install -r requirements-main.txt

# isolated ASR env (qwen-asr pins transformers==4.57.6 which breaks pyannote 4.x)
python3 -m venv .venv-asr
.venv-asr/bin/pip install -r requirements-asr.txt

# one-time gates/downloads
#   https://huggingface.co/pyannote/speaker-diarization-community-1  -> accept terms
hf download Surajgameramp/srota

cp /path/to/.env .env    # SARVAM_API_KEY=... (optional ASR fallback)
```

## Run

```bash
python server.py          # http://127.0.0.1:8765
```

`GET /api/preflight` (also shown in the UI banner) reports anything missing.

## Layout

- `server.py` — FastAPI + single runner thread; stages resume by manifest
- `studio/stages/` — s0 ingest → s1 diarize → s2 denoise → s3 segment → s4 asr → s5 align
- `studio/detectors/` — pluggable; `registry.py` lists them (`laughs` first)
- `workers/asr_worker.py` — runs under `.venv-asr`, one model load per batch
- `jobs/<video_id>/` — master.*, audio/{original,denoised}.wav,
  audio/speakers/*.wav, segments/, peaks/, manifests/*.jsonl

Manifest schemas are documented in `studio/manifests.py` and stay compatible
with the old repo's `podcast_pipeline/common.py`.
