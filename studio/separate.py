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
WINDOW_S = 10.0    # full-audio sweeps use this window


def _decode_16k(wav_bytes: bytes):
    import io

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    g = gcd(16000, sr)
    return resample_poly(data, 16000 // g, sr // g).astype(np.float32)


def run_sam_speaker_full(job_dir: Path, report, speaker: str,
                         model: str = "facebook/sam-audio-base",
                         reranking: int = 1) -> list[str]:
    """Smart full-track clean lane for one speaker:

        where the speaker is ALONE     -> original audio (perfect, free)
        where speakers OVERLAP         -> SAM-separated speaker (GPU, only here)
        where the speaker is silent    -> silence

    SAM only touches the 10s windows containing an overlap involving this
    speaker — the cheapest way to a continuous clean lane, and the original
    (best fidelity) is kept everywhere separation isn't needed."""
    import json

    import numpy as np
    import soundfile as sf

    from .audio import load_audio, render_masked_track
    from .intervals import merge_close

    original = job_dir / "audio" / "original.wav"
    audio = load_audio(original, 16000)
    total_s = len(audio) / 16000

    rows = [json.loads(l) for l in
            (job_dir / "manifests" / "speakers.jsonl").read_text().splitlines() if l.strip()]
    turns = next((r["turns"] for r in rows if r["speaker"] == speaker), None)
    if not turns:
        raise RuntimeError(f"unknown speaker {speaker}")
    my = [(t["start"], t["end"]) for t in turns]
    others = [(t["start"], t["end"]) for r in rows if r["speaker"] != speaker
              for t in r["turns"]]

    # overlaps involving THIS speaker
    ov = []
    for a, b in my:
        for c, d in others:
            s, e = max(a, c), min(b, d)
            if e - s >= 0.1:
                ov.append((s, e))
    ov = merge_close(ov, 0.2)

    # 10s grid windows containing any such overlap
    win_idx = sorted({int(s // WINDOW_S) for s, e in ov}
                     | {int((e - 1e-3) // WINDOW_S) for s, e in ov})
    windows = [(i * WINDOW_S, min((i + 1) * WINDOW_S, total_s)) for i in win_idx]

    # base lane: masked original (speaker-alone regions at full fidelity)
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    name = f"sam_{slug}_clean"
    dest = job_dir / "audio" / f"{name}.wav"
    render_masked_track(audio, my, dest, 16000)
    track, _ = sf.read(dest, dtype="float32")

    if windows:
        report(f"{len(ov)} overlaps -> SAM on {len(windows)} of "
               f"{int(np.ceil(total_s / WINDOW_S))} windows", 0.05)
        import modal
        Separator = modal.Cls.from_name(MODAL_APP, "Separator")
        sep = Separator()
        chunk = job_dir / "_sam_chunk.wav"
        for i, (a, b) in enumerate(windows):
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}",
                 "-t", f"{b - a:.2f}",
                 "-i", str(next(job_dir.glob("master.*"), original)),
                 "-ac", "1", str(chunk)], check=True)
            anchors = speaker_anchors(job_dir, speaker, a, b - a)
            out = sep.separate.remote(chunk.read_bytes(), "speech",
                                      model_name=model, reranking=reranking,
                                      anchors=anchors)
            y = _decode_16k(out["target"])
            i0, i1 = int(a * 16000), min(len(track), int(b * 16000))
            track[i0:i1] = y[: i1 - i0]
            report(f"overlap window {i + 1}/{len(windows)} ({a:.0f}-{b:.0f}s)",
                   0.05 + 0.9 * (i + 1) / len(windows))
        chunk.unlink(missing_ok=True)
    else:
        report("no overlaps involve this speaker — clean lane is just the "
               "masked original", 0.9)

    sf.write(dest, track, 16000, subtype="PCM_16")
    (job_dir / "peaks" / f"{name}.json").unlink(missing_ok=True)
    report(f"clean lane ready: {name} (original + SAM @ {len(windows)} windows)", 1.0)
    return [name]


def speaker_anchors(job_dir: Path, speaker: str, start: float,
                    dur: float) -> list[tuple]:
    """Chunk-relative +/- span prompts from the diarization manifest.

    "+" spans: the target speaker's turns inside the window.
    "-" spans: other speakers' turns MINUS the target's (a span where both
    talk must not be marked negative — that would contradict the positives).
    """
    import json
    rows = [json.loads(l) for l in
            (job_dir / "manifests" / "speakers.jsonl").read_text().splitlines() if l.strip()]
    from .intervals import merge_close, subtract_intervals
    end = start + dur

    def window(turns):
        iv = [(max(t["start"], start) - start, min(t["end"], end) - start)
              for t in turns if t["end"] > start and t["start"] < end]
        return merge_close([(a, b) for a, b in iv if b - a > 0.05], 0.05)

    target = others = None
    others_all = []
    for r in rows:
        if r["speaker"] == speaker:
            target = window(r["turns"])
        else:
            others_all.extend(window(r["turns"]))
    if not target:
        raise RuntimeError(f"{speaker} has no turns in {start:.0f}-{end:.0f}s "
                           f"— pick a window where this speaker talks")
    others = subtract_intervals(merge_close(others_all, 0.05), target)
    anchors = [("+", round(a, 2), round(b, 2)) for a, b in target]
    anchors += [("-", round(a, 2), round(b, 2)) for a, b in others]
    return anchors


def run_sam(job_dir: Path, report, prompt: str, start: float, dur: float,
            model: str = "facebook/sam-audio-base", reranking: int = 1,
            speaker: str | None = None) -> list[str]:
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

    anchors = None
    if speaker:
        anchors = speaker_anchors(job_dir, speaker, start, dur)
        report(f"SAM span-prompt: {sum(1 for a in anchors if a[0] == '+')}+ / "
               f"{sum(1 for a in anchors if a[0] == '-')}- spans", 0.1)

    report(f"SAM-Audio on Modal GPU: {prompt!r}", 0.15)
    import modal
    Separator = modal.Cls.from_name(MODAL_APP, "Separator")
    out = Separator().separate.remote(
        chunk.read_bytes(), prompt, model_name=model, reranking=reranking,
        anchors=anchors)
    chunk.unlink(missing_ok=True)

    report("writing timeline-aligned lanes", 0.85)
    total = sf.info(job_dir / "audio" / "original.wav").frames
    slug = (re.sub(r"[^a-z0-9]+", "_", speaker.lower()) + "_spans") if speaker \
        else re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:24]
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
