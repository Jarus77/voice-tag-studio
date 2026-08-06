"""Detector registry — one entry per pluggable tag detector.

To add a detector: subclass Detector in its own module, register here.
Planned next (order = easiest to verify by ear first, per the podcast
pipeline's quality-test-first protocol):
  whispers   VAD speech + near-zero voiced-frame ratio (physics)
  pauses/hesitates/stammers   pure rules over alignments.jsonl islands
  excited    audeering wav2vec2 arousal/valence, per-speaker percentiles
"""

from __future__ import annotations

from .base import Detector
from .laughs_panns import LaughsPanns

DETECTORS: dict[str, Detector] = {
    "laughs": LaughsPanns(),
}
