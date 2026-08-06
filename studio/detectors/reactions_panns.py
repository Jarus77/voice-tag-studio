"""[sigh] [gasps] — PANNs CNN14 SED, same recipe as laughs.

AudioSet has dedicated Sigh and Gasp classes. These are rarer and subtler
than laughter — expect lower scores; the emit floor stays low and the
audition ear owns the accept decision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detector
from .laughs_panns import PANNS_SR, _get_sed

EMIT = 0.03
CLASS_OF = {"sigh": ["Sigh"], "gasps": ["Gasp"]}


class ReactionsPanns(Detector):
    name = "reactions"
    tags = ["sigh", "gasps"]
    source = "original"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        from panns_inference.config import labels

        from ..audio import load_audio

        idx_of = {tag: [labels.index(c) for c in cls if c in labels]
                  for tag, cls in CLASS_OF.items()}

        report("loading PANNs SED (cpu)", 0.02)
        sed = _get_sed()
        master = next(job_dir.glob("master.*"), None) \
            or (job_dir / "audio" / "original.wav")
        report("decoding master at 32k", 0.05)
        audio = load_audio(master, PANNS_SR)

        out = []
        for i, seg in enumerate(segments):
            i0 = int(seg["start"] * PANNS_SR)
            i1 = min(int(seg["end"] * PANNS_SR), len(audio))
            if i1 - i0 < PANNS_SR // 4:
                continue
            framewise = sed.inference(audio[None, i0:i1])[0]
            frame_s = (i1 - i0) / PANNS_SR / framewise.shape[0]
            for tag, idx in idx_of.items():
                sub = framewise[:, idx]
                score = float(sub.max())
                if score >= EMIT:
                    pk = int(np.argmax(sub.max(axis=1)))
                    out.append({
                        "seg_id": seg["seg_id"], "tag": tag,
                        "score": round(score, 3),
                        "detector": "panns_cnn14_sed",
                        "position_s": round(pk * frame_s, 2),
                        "evidence": {"classes": CLASS_OF[tag], "source": "master@32k"},
                    })
            if (i + 1) % 25 == 0 or i + 1 == len(segments):
                report(f"scanned {i + 1}/{len(segments)}, {len(out)} candidates",
                       0.05 + 0.93 * (i + 1) / len(segments))
        return out
