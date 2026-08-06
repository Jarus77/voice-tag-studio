"""[laughs] detector — PANNs CNN14 sound-event detection (AudioSet).

Frame-level scores for the laughter family of AudioSet classes. Language-
independent, easiest tag to verify by ear — which is why it's first.

Mechanics: decode the ARCHIVAL MASTER once at 32 kHz (PANNs' native rate;
upsampling the 16 k working wav would discard content), slice per segment,
run SED, take the frame-max over laugh classes. Low emit threshold (0.20):
the UI sorts by score and the ear decides; thresholds move after audition.

panns_inference is 2021-era: torch.load needs weights_only=False under
torch 2.x, and device support is cuda-or-cpu only (cpu here; fine for CNN14).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detector

PANNS_SR = 32000
# Calibration observed on real data (2026-08-06):
#   overt/audience laughter  -> 0.32-0.43
#   conversational chuckles  -> 0.05-0.18
#   laugh-free speech        -> ~0.002
# Emit low and let the audition ear decide (UI sorts by score); accept
# thresholds move AFTER per-tag precision estimates, never before.
EMIT_THRESHOLD = 0.05
LAUGH_CLASSES = ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]

_sed = None  # lazy singleton — checkpoint load is ~seconds


def _get_sed():
    global _sed
    if _sed is None:
        import torch
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        try:
            from panns_inference.inference import SoundEventDetection
            _sed = SoundEventDetection(device="cpu")
        finally:
            torch.load = _orig
    return _sed


class LaughsPanns(Detector):
    name = "laughs"
    tags = ["laughs"]
    source = "original"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        from panns_inference.config import labels

        from ..audio import load_audio

        cls_idx = [labels.index(c) for c in LAUGH_CLASSES if c in labels]

        report("loading PANNs SED (cpu)", 0.02)
        sed = _get_sed()

        master = next(job_dir.glob("master.*"), None) \
            or (job_dir / "audio" / "original.wav")
        report("decoding master at 32k", 0.05)
        audio = load_audio(master, PANNS_SR)

        out_rows = []
        for i, seg in enumerate(segments):
            i0 = int(seg["start"] * PANNS_SR)
            i1 = min(int(seg["end"] * PANNS_SR), len(audio))
            if i1 - i0 < PANNS_SR // 4:
                continue
            clip = audio[i0:i1]
            framewise = sed.inference(clip[None, :])[0]      # (frames, 527)
            frame_s = (i1 - i0) / PANNS_SR / framewise.shape[0]
            laugh = framewise[:, cls_idx]                     # (frames, n_cls)
            peak_frame = int(np.argmax(laugh.max(axis=1)))
            score = float(laugh.max())
            if score >= EMIT_THRESHOLD:
                per_class = {c: round(float(framewise[:, labels.index(c)].max()), 3)
                             for c in LAUGH_CLASSES if c in labels}
                out_rows.append({
                    "seg_id": seg["seg_id"],
                    "tag": "laughs",
                    "score": round(score, 3),
                    "detector": "panns_cnn14_sed",
                    "position_s": round(peak_frame * frame_s, 2),
                    "evidence": {"class_scores": per_class,
                                 "source": "master@32k"},
                })
            if (i + 1) % 10 == 0 or i + 1 == len(segments):
                report(f"scanned {i + 1}/{len(segments)} segments, "
                       f"{len(out_rows)} candidates",
                       0.05 + 0.93 * (i + 1) / len(segments))
        return out_rows
