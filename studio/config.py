"""Central config for voice-tag-studio."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
STATIC_DIR = ROOT / "static"
ASR_VENV_PY = ROOT / ".venv-asr" / "bin" / "python"
ASR_WORKER = ROOT / "workers" / "asr_worker.py"

PORT = int(os.environ.get("TAG_STUDIO_PORT", "8765"))

# ---- audio conventions (carried over from the voice pipeline) ----
SR = 16000                # internal working rate; master download kept untouched
MIN_SEG_S = 2.0
MAX_SEG_S = 20.0
MERGE_GAP_S = 0.30        # merge clean runs separated by less than this
CROSSTALK_GUARD_S = 0.15  # widen other-speaker turns by this before subtracting
PAD_S = 0.08              # breathing room re-added after energy trim
FRAME_S = 0.010
MASK_FADE_S = 0.020       # linear fades at speaker-mask edges (kill clicks)

# ---- models ----
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
VAD_MODEL = "pyannote/segmentation-3.0"   # for MMS-alignment speech intervals
ASR_MODEL = "Surajgameramp/srota"
MAX_SPEAKERS = 6

# ---- alignment (stage5b recipe) ----
MIN_ISLAND_S = 0.30
FILLER_MAX_S = 1.50
WORD_PAD_S = 0.12
MIN_SPEECH_FRAMES = 3

# ---- policy toggles ----
# Detectors read the ORIGINAL audio by default: demucs strips/attenuates
# laughter and other non-speech vocal events -- exactly what we tag.
DETECTOR_SOURCE_DEFAULT = "original"
# v1: diarization + ASR also run on original (podcast speech is usually clean
# enough; denoised is a listening/QA artifact).
NO_CUT_ZONE_PAD_S = 0.5   # protection window around detected tag events

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v3"


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
