"""Audio I/O helpers: ffmpeg decode, probe, cutting, peaks, masked tracks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MASK_FADE_S, SR


def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    """Decode any audio to mono float32 via ffmpeg pipe."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def to_wav(src: Path, dest: Path, sr: int = SR) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
           "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True, capture_output=True)


def cut_segment(src: Path, start: float, dur: float, dest: Path,
                sr: int = SR) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", str(src), "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception:
        return False


def probe(path: Path) -> dict:
    """ffprobe a single file. Returns {} on failure (corrupt file)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=sample_rate,channels,bit_rate",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return {}
        data = json.loads(out.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        return {
            "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            "channels": int(stream["channels"]) if stream.get("channels") else None,
            "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
            "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        }
    except Exception:
        return {}


def compute_peaks(wav: Path, n_bins: int = 2000) -> dict:
    """Min/max envelope for canvas waveform rendering."""
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    if n == 0:
        return {"duration": 0.0, "bins": []}
    hop = max(1, n // n_bins)
    usable = (n // hop) * hop
    frames = audio[:usable].reshape(-1, hop)
    bins = np.stack([frames.min(axis=1), frames.max(axis=1)], axis=1)
    return {
        "duration": round(n / sr, 3),
        "bins": [[round(float(a), 4), round(float(b), 4)] for a, b in bins],
    }


def peaks_cached(wav: Path, cache_dir: Path, n_bins: int = 2000) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (wav.stem + ".json")
    if cache.exists() and cache.stat().st_mtime >= wav.stat().st_mtime:
        return json.loads(cache.read_text())
    data = compute_peaks(wav, n_bins)
    cache.write_text(json.dumps(data))
    return data


def render_masked_track(audio: np.ndarray, keep: list[tuple], dest: Path,
                        sr: int = SR) -> None:
    """Full-length track with everything outside `keep` intervals silenced.

    Timeline is preserved (transcript seek stays trivial; diarization errors
    are audible in context). 20 ms linear fades at mask edges kill clicks.
    """
    mask = np.zeros(len(audio), dtype=np.float32)
    fade = max(1, int(MASK_FADE_S * sr))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    for s, e in keep:
        i0, i1 = max(0, int(s * sr)), min(len(audio), int(e * sr))
        if i1 <= i0:
            continue
        mask[i0:i1] = 1.0
        # fade-in / fade-out shoulders (clipped at array edges)
        a0 = max(0, i0 - fade)
        if i0 > a0:
            mask[a0:i0] = np.maximum(mask[a0:i0], ramp[-(i0 - a0):])
        b1 = min(len(audio), i1 + fade)
        if b1 > i1:
            mask[i1:b1] = np.maximum(mask[i1:b1], ramp[::-1][: b1 - i1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio * mask, sr, subtype="PCM_16")
