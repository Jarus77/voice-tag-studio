"""[whispers] — physics, language-independent.

Whispered speech has energy but (almost) no glottal periodicity: speech-level
RMS with near-zero voiced-frame ratio. Uses the shared prosody feature cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detector
from .prosody import features

VOICED_MAX = 0.25    # normal speech is ~0.4-0.8 voiced; whisper ~0-0.15
RMS_MIN_PCTL = 20    # must still be audibly speaking (not just silence)


class WhispersVoicing(Detector):
    name = "whispers"
    tags = ["whispers"]
    source = "original"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        feats = features(job_dir, segments, report)
        rms_all = [f["rms"] for f in feats.values()
                   if isinstance(f.get("rms"), (int, float)) and f["rms"] > 0]
        rms_floor = np.percentile(rms_all, RMS_MIN_PCTL) if rms_all else 0.0

        out = []
        for seg in segments:
            f = feats.get(seg["seg_id"], {})
            vr, rms = f.get("voiced_ratio"), f.get("rms")
            if vr is None or rms is None or rms < rms_floor:
                continue
            if vr <= VOICED_MAX:
                out.append({
                    "seg_id": seg["seg_id"], "tag": "whispers",
                    "score": round(float(1.0 - vr), 3),
                    "detector": "voicing_ratio",
                    "position_s": None,   # utterance-level quality
                    "evidence": {"voiced_ratio": vr, "rms": rms},
                })
        report(f"{len(out)} whisper candidates", 1.0)
        return out
