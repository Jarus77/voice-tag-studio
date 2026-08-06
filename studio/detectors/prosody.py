"""Shared per-segment prosody features, cached per job.

whispers + tones + states all need pitch/energy stats; pyin is the slow part,
so compute once and cache to manifests/prosody_features.json.

Per segment: f0_mean, f0_std (semitone-ish via log2), voiced_ratio,
rms, rate_wps (words/sec from alignment when available).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import SR
from ..manifests import read_jsonl

CACHE = "manifests/prosody_features.json"


def features(job_dir: Path, segments: list[dict], report) -> dict[str, dict]:
    cache_path = job_dir / CACHE
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # errored entries are retried, not kept forever
    cached = {k: v for k, v in cached.items() if "error" not in v}
    todo = [s for s in segments if s["seg_id"] not in cached]
    if not todo:
        return cached

    import librosa
    import soundfile as sf

    words_of = {r["seg_id"]: r.get("words") or [] for r in
                read_jsonl(job_dir / "manifests" / "alignments.jsonl")}

    for i, seg in enumerate(todo):
        try:
            x, sr = sf.read(job_dir / seg["wav"], dtype="float32")
            f0, voiced, _ = librosa.pyin(
                x, fmin=60, fmax=400, sr=sr,
                frame_length=1024, hop_length=320, fill_na=np.nan)
            v = np.asarray(voiced, dtype=bool)
            f0v = f0[v & np.isfinite(f0)]
            # energy-active frames (exclude trailing/leading quiet)
            frame = 320
            n = len(x) // frame
            rms = np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
            active = rms > max(np.percentile(rms, 10) * 3, rms.max() * 0.05)
            # pyin and the RMS grid have slightly different frame counts
            # (window padding) — align before masking
            m = min(len(v), len(active))
            va, aa = v[:m], active[:m]
            st = np.log2(f0v) * 12 if len(f0v) else np.array([])
            nw = len(words_of.get(seg["seg_id"], []))
            cached[seg["seg_id"]] = {
                "f0_mean": round(float(f0v.mean()), 2) if len(f0v) else None,
                "f0_std_st": round(float(st.std()), 3) if len(st) > 3 else None,
                "voiced_ratio": round(float(va[aa].mean()), 3) if aa.any() else 0.0,
                "rms": round(float(rms[active].mean()), 5) if active.any() else 0.0,
                "rate_wps": round(nw / seg["dur"], 2) if nw else None,
            }
        except Exception as e:
            cached[seg["seg_id"]] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            report(f"prosody features {i + 1}/{len(todo)}",
                   0.9 * (i + 1) / len(todo))
            cache_path.write_text(json.dumps(cached))
    cache_path.write_text(json.dumps(cached))
    return cached


def speaker_percentile(values: dict[str, float], x: float) -> float:
    """Percentile (0-1) of x within a speaker's own value distribution."""
    arr = np.array([v for v in values.values() if v is not None])
    if len(arr) < 10:
        return 0.5
    return float((arr < x).mean())
