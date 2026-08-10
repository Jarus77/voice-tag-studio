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
# 1.0s (was 2.0): with word-aware cutting, text/audio match is guaranteed by
# construction, so short standalone utterances ("हेलो", "हाँ जी") — which a
# conversational agent must produce constantly — can safely enter the dataset.
MIN_SEG_S = 1.0
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
# A clip's first/last word is called "clipped" only if its alignment runs
# essentially INTO the edge. energy_trim leaves PAD_S (0.08s) of breathing
# room, so healthy clips sit ~0.06s off the edge — 0.05 flagged 69/101 clips
# on real data (measured 2026-08-10); 0.015 flags ~14, a believable rate.
CLIPPED_EDGE_S = 0.015

# ---- policy toggles ----
# Detectors read the ORIGINAL audio by default: demucs strips/attenuates
# laughter and other non-speech vocal events -- exactly what we tag.
DETECTOR_SOURCE_DEFAULT = "original"
# v1: diarization + ASR also run on original (podcast speech is usually clean
# enough; denoised is a listening/QA artifact).
NO_CUT_ZONE_PAD_S = 0.5   # protection window around detected tag events

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = "saaras:v3"

# ---- export ----
# Per-tag accept thresholds: a candidate becomes a TAG in the exported text
# only at/above these. PROVISIONAL — calibrate each by auditioning ~25 hits
# and moving the number until hand-precision is ~90% (precision over recall:
# a wrong tag is a text/audio mismatch, i.e. hallucination fuel; a missed tag
# is merely unused data).
TAG_THRESHOLDS = {
    # reactions — PANNs probabilities
    "laughs": 0.20, "sigh": 0.10, "gasps": 0.10,
    # beats — rule scores (pauses: 0.25 == a 0.5s gap)
    "pauses": 0.25, "hesitates": 0.50, "stammers": 0.50,
    # effort / tone / states — percentile confidences
    "whispers": 0.60, "flatly": 0.60, "cheerfully": 0.60,
    "excited": 0.60, "nervous": 0.60, "frustrated": 0.60,
    "sorrowful": 0.60, "calm": 0.60,
}

# Tags that describe the WHOLE utterance render at the start of the line;
# everything else renders inline at its detected position.
UTTERANCE_TAGS = {"whispers", "flatly", "cheerfully", "deadpan", "playfully",
                  "excited", "nervous", "frustrated", "sorrowful", "calm",
                  "resigned"}

# Clip filters — all OFF by default (user decision 2026-08-08: keep every clip
# and ablate later). Each row carries its flags so any subset is reproducible.
EXPORT_EXCLUDE = {
    "impure": False,        # purity below PURITY_OK (flagged, kept)
    "clipped": True,        # first/last word cut in half = text/audio mismatch
    "separated": True,      # SepFormer-reconstructed audio has audible
                            # artifacts on telephony — transcripts/tags keep
                            # the rescue value; the AUDIO never trains
    "major_gap": True,      # >1.5s audible speech with NO text = mismatch
    "align_error": True,    # alignment failed — correspondence unverifiable
    "no_text": True,        # a clip with no transcript can never be a row
}


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
