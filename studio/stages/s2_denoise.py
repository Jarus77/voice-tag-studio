"""Stage 2 — Denoise: demucs htdemucs vocals stem -> audio/denoised.wav.

POLICY: denoised audio is for LISTENING/QA only. The original stays the
dataset master (enhancement artifacts must not leak into TTS training clips),
and detectors read the original too — htdemucs strips/attenuates laughter and
other non-speech vocal events, exactly what we tag.

Input is the archival master (best quality in -> best separation out).
Fallback on demucs failure: ffmpeg afftdn spectral denoise.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ..audio import to_wav
from ..config import SR
from ..manifests import write_json


def run(job_dir: Path, report) -> None:
    master = next(job_dir.glob("master.*"), None)
    src = master or (job_dir / "audio" / "original.wav")
    if not src.exists():
        raise RuntimeError("no master/original audio — run ingest first")
    dest = job_dir / "audio" / "denoised.wav"
    tmp = job_dir / "_demucs"
    method = "demucs:htdemucs/vocals"

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        report(f"demucs htdemucs vocals on {device} (this can take a while)", 0.1)
        cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
               "-n", "htdemucs", "-d", device, "-o", str(tmp), str(src)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(f"demucs: {(proc.stderr or proc.stdout)[-300:]}")
        vocals = next(tmp.glob("htdemucs/*/vocals.wav"), None)
        if vocals is None:
            raise RuntimeError("demucs produced no vocals.wav")
        report("resampling vocals to 16k mono", 0.9)
        to_wav(vocals, dest, SR)
    except Exception as e:
        report(f"demucs failed ({type(e).__name__}) — ffmpeg afftdn fallback", 0.5)
        method = "ffmpeg:afftdn"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
               "-af", "afftdn=nf=-25", "-ac", "1", "-ar", str(SR),
               "-c:a", "pcm_s16le", str(dest)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    write_json(job_dir / "manifests" / "denoise.json",
               {"method": method, "source": src.name, "wav": "audio/denoised.wav"})
