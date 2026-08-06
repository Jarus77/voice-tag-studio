"""Tone cues — [flatly] [cheerfully], per-speaker percentile thresholds.

Anything prosodic must be RELATIVE to the speaker's own baseline, never
absolute (hard-won voice-project rule). flatly = bottom-decile pitch
variability; cheerfully = top-decile pitch level AND variability.

deadpan/playfully are deliberately NOT emitted here — prosody stats can't
carry them; they belong to the audio-LLM judge tier.
"""

from __future__ import annotations

from pathlib import Path

from .base import Detector
from .prosody import features, speaker_percentile

FLAT_PCTL = 0.10
CHEER_PCTL = 0.90
MIN_SEGS_PER_SPK = 20   # percentiles are meaningless on tiny samples


class TonesProsody(Detector):
    name = "tones"
    tags = ["flatly", "cheerfully"]
    source = "original"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        feats = features(job_dir, segments, report)
        by_spk: dict[str, list[dict]] = {}
        for seg in segments:
            by_spk.setdefault(seg["speaker"], []).append(seg)

        out = []
        for spk, segs in by_spk.items():
            if len(segs) < MIN_SEGS_PER_SPK:
                continue
            std_of = {s["seg_id"]: feats.get(s["seg_id"], {}).get("f0_std_st")
                      for s in segs}
            mean_of = {s["seg_id"]: feats.get(s["seg_id"], {}).get("f0_mean")
                       for s in segs}
            for s in segs:
                f = feats.get(s["seg_id"], {})
                std, mean, vr = f.get("f0_std_st"), f.get("f0_mean"), f.get("voiced_ratio")
                if std is None or mean is None:
                    continue
                if vr is not None and vr < 0.25:
                    continue   # whisper territory, not "flat" speech
                p_std = speaker_percentile(std_of, std)
                p_mean = speaker_percentile(mean_of, mean)
                if p_std <= FLAT_PCTL:
                    out.append({
                        "seg_id": s["seg_id"], "tag": "flatly",
                        "score": round(1.0 - p_std / FLAT_PCTL * 0.5, 3),
                        "detector": "prosody_percentile",
                        "position_s": None,
                        "evidence": {"f0_std_st": std, "pctl_in_speaker": round(p_std, 3)},
                    })
                elif p_mean >= CHEER_PCTL and p_std >= 0.60:
                    out.append({
                        "seg_id": s["seg_id"], "tag": "cheerfully",
                        "score": round(0.5 + (p_mean - CHEER_PCTL) / (1 - CHEER_PCTL) * 0.5, 3),
                        "detector": "prosody_percentile",
                        "position_s": None,
                        "evidence": {"f0_mean": mean, "f0_std_st": std,
                                     "pctl_mean": round(p_mean, 3),
                                     "pctl_std": round(p_std, 3)},
                    })
        report(f"{len(out)} tone candidates", 1.0)
        return out
