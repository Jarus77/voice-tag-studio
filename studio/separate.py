"""SAM-Audio separation, driven from the server via the deployed Modal app.

Cuts a chunk from the archival master, sends it to the `tag-studio-sam-audio`
Modal deployment with a text prompt, and writes the target/residual back as
FULL-LENGTH timeline-aligned lanes (silence outside the chunk) so they appear
in the Listen section next to original/denoised.

Separated audio is a listening/detector aid — never dataset master.
"""

from __future__ import annotations

import io
import re
import subprocess
from math import gcd
from pathlib import Path

MODAL_APP = "tag-studio-sam-audio"
MAX_DUR_S = 30.0   # SAM-Audio works best near 10 s (its training length)


def run_sam(job_dir: Path, report, prompt: str, start: float, dur: float,
            model: str = "facebook/sam-audio-base", reranking: int = 1) -> list[str]:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    dur = min(float(dur), MAX_DUR_S)
    master = next(job_dir.glob("master.*"), None) \
        or (job_dir / "audio" / "original.wav")
    if not master.exists():
        raise RuntimeError("no master/original audio — run ingest first")

    report(f"cutting {dur:.0f}s @ {start:.0f}s", 0.05)
    chunk = job_dir / "_sam_chunk.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
         "-i", str(master), "-ac", "1", str(chunk)], check=True)

    report(f"SAM-Audio on Modal GPU: {prompt!r}", 0.15)
    import modal
    Separator = modal.Cls.from_name(MODAL_APP, "Separator")
    out = Separator().separate.remote(
        chunk.read_bytes(), prompt, model_name=model, reranking=reranking)
    chunk.unlink(missing_ok=True)

    report("writing timeline-aligned lanes", 0.85)
    total = sf.info(job_dir / "audio" / "original.wav").frames
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:24]
    written = []
    for kind in ("target", "residual"):
        data, sr = sf.read(io.BytesIO(out[kind]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        g = gcd(16000, sr)
        y = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
        track = np.zeros(total, dtype=np.float32)
        i0 = int(start * 16000)
        i1 = min(total, i0 + len(y))
        track[i0:i1] = y[: i1 - i0]
        name = f"sam_{slug}_{kind}"
        sf.write(job_dir / "audio" / f"{name}.wav", track, 16000, subtype="PCM_16")
        # stale peaks cache would hide the new audio
        peak = job_dir / "peaks" / f"{name}.json"
        peak.unlink(missing_ok=True)
        written.append(name)
    report(f"lanes ready: {', '.join(written)}", 1.0)
    return written
