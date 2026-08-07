"""Stage 0 — Ingest: YouTube URL -> archival master + 16k working wav.

The bestaudio download (m4a/webm, 44.1/48 kHz) is the MASTER and is never
modified — 16k is for models, not archival. audio/original.wav (16 kHz mono
PCM_16) is the working copy every downstream stage reads.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..audio import probe, to_wav
from ..config import SR
from ..manifests import write_json


def run(job_dir: Path, report, url: str) -> None:
    """url == "" means uploaded-audio mode: master.* was saved by the upload
    endpoint; metadata comes from ffprobe instead of yt-dlp."""
    vid = job_dir.name
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    master = next(job_dir.glob("master.*"), None)
    if not url:
        if master is None:
            raise RuntimeError("upload job has no master file")
        meta = {"title": job_dir.name, "channel": "upload",
                "duration": (probe(master) or {}).get("duration_s")}
    else:
        report("fetching metadata", 0.05)
        meta_raw = subprocess.run(
            ["yt-dlp", "--no-playlist", "-J", url],
            capture_output=True, text=True, timeout=120)
        if meta_raw.returncode != 0:
            raise RuntimeError(f"yt-dlp metadata: {meta_raw.stderr[-200:]}")
        meta = json.loads(meta_raw.stdout)

        if master is None:
            report(f"downloading audio ({meta.get('duration', '?')}s)", 0.15)
            dl = subprocess.run(
                ["yt-dlp", "--no-playlist", "-f", "bestaudio",
                 "-o", str(job_dir / "master.%(ext)s"), url],
                capture_output=True, text=True, timeout=1800)
            if dl.returncode != 0:
                raise RuntimeError(f"yt-dlp download: {dl.stderr[-200:]}")
            master = next(job_dir.glob("master.*"), None)
            if master is None:
                raise RuntimeError("download produced no master file")

    report("decoding to 16k mono wav", 0.7)
    wav = audio_dir / "original.wav"
    to_wav(master, wav, SR)

    report("probing", 0.9)
    write_json(job_dir / "manifests" / "stage0_video.json", {
        "video_id": vid,
        "url": url,
        "title": meta.get("title", ""),
        "channel": meta.get("channel") or meta.get("uploader", ""),
        "duration_s": meta.get("duration"),
        "master_path": master.name,
        "master_probe": probe(master),
        "wav_path": "audio/original.wav",
        "sample_rate": SR,
    })
