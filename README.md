# voice-tag-studio

A local data-annotation workbench for building **emotion-tagged TTS datasets** from
Hindi/Hinglish audio — podcasts, calls, YouTube, anything with speech.

Feed it audio, and it produces training rows of the form:

```
speaker: जो पिघले न [hesitates] देखा जाए तो [pauses] पर आप तो मतलब आप बोलते हो
```

…paired with the exact voice clip. Run it in a browser to inspect every decision
by ear, or headless over a folder of files.

![voice-tag-studio](docs/screenshot.png)

*The pipeline runs stage by stage — you click each one. Below it, every version of
the audio on one timeline: the original, a SepFormer-cleaned lane per speaker, and
the raw diarized lanes. Amber marks show where speakers overlap; pink diamonds are
tag hits.*

---

## Pipeline

```
ingest → diarize → [clean] → asr → align → segment → detectors → export
```

Transcription happens **before** cutting: each speaker's lane is transcribed in
~60 s chunks and force-aligned, and clips are then cut **only in verified gaps
between words**. Text↔audio match holds by construction — a boundary can never
truncate a word, because word positions are known before any cut is made.

| stage | what it does | model |
|---|---|---|
| **ingest** | YouTube URL or uploaded file → archival master + 16 kHz working wav | yt-dlp, ffmpeg |
| **diarize** | who speaks when | `pyannote/speaker-diarization-community-1` (MPS) |
| **clean** *(optional)* | rescues overlapped speech instead of discarding it | SepFormer + ECAPA purity gate |
| **asr** | verbatim Hinglish lane transcript in ~60 s chunks, **fillers preserved** | Gemini 2.5 Flash (or srota) |
| **align** | word-level timestamps on the lane, unaligned-speech "islands" | torchaudio `MMS_FA` + uroman |
| **segment** | cuts 1–20 s clips **only in verified word gaps**; clip text is derived from the aligned words inside it. Clips cut *around* rescued overlap and alignment glitches; solo windows are cut from the **original recording** (continuous room tone, no processing artifacts) | word timings, no model |
| **detectors** | emotion tag candidates | see below |
| **export** | `dataset.jsonl` — clip + tagged text + speaker × tag matrix | — |

Nothing runs automatically except ingest: **every stage runs when you click it**,
so you can inspect the output before moving on. **Run remaining ▸** queues every
not-yet-done stage in order — including all detectors before export, so one click
ends in a tagged dataset.

---

## Emotion tags

| family | tags | detector |
|---|---|---|
| **reactions** | `[laughs]` `[sigh]` `[gasps]` | PANNs CNN14 sound-event detection (AudioSet) |
| **beats** | `[pauses]` `[silence]` `[hesitates]` `[stammers]` | rules over forced alignment — deterministic, free. `[pauses]` (0.5–1.5 s gap) and `[silence]` (≥1.5 s) are also derived automatically at export, so in-clip silence is always in the text |
| **vocal effort** | `[whispers]` | voiced-frame ratio (physics, language-independent) |
| **tone** | `[flatly]` `[cheerfully]` | per-speaker pitch percentiles |
| **states** | `[excited]` `[nervous]` `[frustrated]` `[sorrowful]` `[calm]` | audeering wav2vec2 arousal/valence, per-speaker percentiles |

Positioned tags (reactions, beats) render **inline** at the moment they occur;
utterance-level tags (tone, effort, states) render at the **start** of the line.

**These are candidates, not labels.** Reliability drops down the table: beats are
deterministic timing maths, states use an English-trained model on Hindi and are
marked weak evidence. Audition before trusting — thresholds live in
`studio/config.py::TAG_THRESHOLDS` and are meant to be tuned by ear.

---

## Setup

Requires Python 3.11+, `ffmpeg`, and a Hugging Face account.

```bash
pip install -r requirements-main.txt

# ASR worker needs its own venv (qwen-asr pins transformers 4.57, which
# conflicts with pyannote 4.x). Only needed if you use the srota engine.
python3 -m venv .venv-asr && .venv-asr/bin/pip install -r requirements-asr.txt

# one-time: accept the gated diarization model
open https://huggingface.co/pyannote/speaker-diarization-community-1
hf auth login

cp .env.example .env      # add GEMINI_API_KEY (and optionally SARVAM_API_KEY)
```

The **Preflight** banner in the UI reports anything missing, with the fix.

**Optional — GPU acceleration** (each a one-time deploy; every stage
auto-detects its app and falls back to local compute when missing):

```bash
modal deploy modal_sepformer.py   # clean: SepFormer on T4 — minutes, not hours
modal deploy modal_align.py       # align: MMS forced alignment on T4
```

Separated windows are cached per job, so additional speakers reuse the first
speaker's separation work. Force backends with `VTS_SEPFORMER_BACKEND` /
`VTS_ALIGN_BACKEND` = `cpu|modal` (default: auto). ASR concurrency is tunable
with `VTS_ASR_CONCURRENCY` (default 16 parallel Gemini calls).

---

## Use it — browser

```bash
python server.py          # http://127.0.0.1:8765
```

1. **Upload audio** (short clips are ideal) or paste a **YouTube URL** → ingest runs.
2. Click **diarize**. Speaker lanes appear — click a lane name to hear only that voice,
   or name it (`agent`, `khushi`) since `SPEAKER_00` is an arbitrary label.
3. *(optional)* Click **clean** on a speaker to rescue their overlapped speech.
4. Click **asr** → **align** → **segment**. The Segments view **is the training
   data**: each row is one clip beside its exact text (karaoke-highlighted as it
   plays); dimmed rows are excluded by the export gate, with the reason.
5. **Tags** section: run one detector and audition each hit with ▶ ±2s, or
   **Run all ▸** for every detector (re-run it after re-cutting segments).
   Tags appear inline in the transcript at their exact position.
6. Click **export** → `jobs/<id>/manifests/dataset.jsonl`.

![segments and tags](docs/transcript-tag.png)

*Each clip shows its waveform beside its words. Tags sit inline exactly where they
occur — `[hesitates] 0.58s`, `[pauses] 0.72s`, `[stammers] "एक"` — with the
quantity you can judge by ear rather than an opaque score. Words highlight
karaoke-style during playback.*

**Export gate** (`studio/config.py::EXPORT_EXCLUDE`): clips whose text↔audio
correspondence is broken or unverifiable — no text, edge word cut in half,
SepFormer-reconstructed audio, major alignment gap, alignment failure — stay
visible in the UI but never become dataset rows. `impure` (voiceprint below
0.55) is kept and flagged: it's a suspicion, not a proven mismatch.

**Quality flags** shown per clip — nothing is silently dropped:
`rescued N%` (share of separated audio) · `impure 0.54` (voice mismatch) ·
`✂ start/end` (word cut in half) · `deva/lat %` (Hinglish script mix) ·
`digits_uncertain` (unaligned audio explained by digits).

---

## Use it — batch

```bash
# a folder of calls, with detectors and overlap rescue
python run_batch.py calls/ --clean --detectors beats,laughs,tones,states

# YouTube
python run_batch.py "https://youtu.be/XXXXXXXXXXX" --detectors beats

# merge every processed job into one portable corpus
python run_batch.py --collect-only --out corpus/
```

`--collect-only --out` copies the clips beside a combined `dataset.jsonl` and
prints the **speaker × tag matrix** — the artifact that tells you which
(speaker, tag) cells are interpolation at inference time and which are
extrapolation.

---

## Output

`dataset.jsonl`, one row per clip:

```json
{
  "sample_id": "EOKZrphv3QI_SPEAKER_01_0007",
  "speaker": "SPEAKER_01",
  "line": "SPEAKER_01: जो पिघले न [hesitates] देखा जाए तो [pauses] पर आप तो...",
  "text": "जो पिघले न [hesitates] देखा जाए तो [pauses] पर आप तो...",
  "text_plain": "जो पिघले न देखा जाए तो पर आप तो...",
  "tags": ["hesitates", "pauses"],
  "wav": "segments/SPEAKER_01/EOKZrphv3QI_SPEAKER_01_0007.wav",
  "dur": 17.9, "split": "train",
  "source": "original", "separated_frac": 0.0,
  "purity": 0.83, "align_status": "filler_suspect", "clipped": []
}
```

Every row carries its provenance and quality flags, so any filtered subset is
reproducible without re-running anything. Clips are 16 kHz mono PCM_16; the
original best-quality download is always kept as `master.*`.

---

## Design rules

Carried over from the production voice pipeline this grew out of:

- **Precision over recall.** A wrong tag is a text↔audio mismatch — hallucination
  fuel. A missed tag is merely unused data.
- **Never train on two voices.** Other speakers' turns are subtracted with a guard
  band; overlapped seconds never enter a clip.
- **Train on the raw recording.** Solo clip windows are cut from the original
  audio — diarization, separation and alignment only *decide* where to cut and
  what the text is; they never process what the model hears.
- **Silence must be in the text.** In-clip word gaps become `[pauses]` (0.5–1.5 s)
  or `[silence]` (≥1.5 s) automatically at export — untagged silence is a
  text↔audio mismatch.
- **Distrust absurd timings.** A single aligned word longer than 2 s is an
  alignment glitch; its span is unusable and clips cut around it.
- **Separated audio is a working surface, never training audio.** SepFormer's
  reconstruction feeds ASR and alignment (so transcripts survive overlap), but
  clips are cut *around* rescued regions — every trained second is original audio.
- **Per-speaker percentiles** for anything prosodic — never absolute thresholds.
- **Split by video, never by segment** — same-conversation leakage is invisible
  and fatal.
- **Neutral rows matter.** A speaker needs untagged rows for tags to mean anything.

`DETECTOR_EVAL_PLAN.md` tracks the model bake-offs still to run.
