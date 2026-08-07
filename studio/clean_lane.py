"""Overlap-rescue v2: clean per-speaker lanes with a measurable quality gate.

    detection   tier 1: sweep over pyannote turns (+0.4s dilation)
                tier 2: acoustic — segmentation-3.0 frame-level "≥2 speakers"
                (catches backchannels the diarizer never emitted as turns)
    eliminate   2 diarized speakers -> SepFormer (local, discriminative, free)
                else               -> SAM-Audio span-prompted (Modal GPU)
    stitch      original where solo · separated at suspect windows · silence
                elsewhere · 20ms crossfades at every paste edge
    verify      ecapa purity scan vs the speaker's own centroid; per-window
                scores saved to manifests/clean_purity_<spk>.json

Purity-gated windows may later become TRAINING-data candidates (rows will
carry source="separated" so they can always be excluded); ungated separated
audio never leaves the listening tier.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

WINDOW_S = 10.0
XFADE_S = 0.02
PAD_S = 0.4           # dilation of other-speaker turns (boundary bleed)
PURITY_OK = 0.55      # cosine-to-centroid above this = target's voice

_ecapa = None
_sepformer = None
_seg_inference = None


# ---------------- models (local, lazy) ----------------

def _get_ecapa():
    global _ecapa
    if _ecapa is None:
        from speechbrain.inference.speaker import EncoderClassifier
        # speechbrain 1.1.0 breaks on mps (missing device_type attr) — cpu is
        # plenty for 1s embedding windows
        _ecapa = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
    return _ecapa


def _embed(x: np.ndarray) -> np.ndarray:
    import torch
    enc = _get_ecapa()
    with torch.inference_mode():
        e = enc.encode_batch(torch.from_numpy(x)[None]).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def _get_sepformer():
    global _sepformer
    if _sepformer is None:
        from speechbrain.inference.separation import SepformerSeparation
        _sepformer = SepformerSeparation.from_hparams(
            "speechbrain/sepformer-whamr16k", run_opts={"device": "cpu"})
    return _sepformer


def acoustic_overlaps(wav: Path, device: str = "cpu") -> list[tuple]:
    """Frame-level ≥2-speakers regions from pyannote segmentation-3.0 —
    independent of turn clustering, so short backchannels are caught."""
    global _seg_inference
    from .intervals import merge_close
    if _seg_inference is None:
        import torch
        from huggingface_hub import get_token
        from pyannote.audio import Inference, Model
        seg = Model.from_pretrained("pyannote/segmentation-3.0",
                                    use_auth_token=get_token())
        _seg_inference = Inference(seg, duration=10.0, step=10.0,
                                   device=torch.device(device))
    out = _seg_inference(str(wav))
    data = np.asarray(out.data)
    if data.ndim == 2:
        data = data[None]
    n_frames = data.shape[1]
    frame_s = 10.0 / n_frames
    iv = []
    for w, win in enumerate(data):
        act = ((win >= 0.5).sum(axis=-1) >= 2)
        start = None
        for f, v in enumerate(np.append(act, 0)):
            if v and start is None:
                start = f
            elif not v and start is not None:
                if f - start >= 3:      # <30ms runs are powerset flicker
                    iv.append((w * 10.0 + start * frame_s,
                               w * 10.0 + f * frame_s))
                start = None
    return merge_close(iv, 0.2)


# ---------------- routing ----------------

def suspect_regions(job_dir: Path, speaker: str, report) -> tuple:
    """tier1 ∪ tier2 overlap regions involving the target speaker."""
    from .intervals import merge_close
    rows = [json.loads(l) for l in
            (job_dir / "manifests" / "speakers.jsonl").read_text().splitlines() if l.strip()]
    turns = next((r["turns"] for r in rows if r["speaker"] == speaker), None)
    if not turns:
        raise RuntimeError(f"unknown speaker {speaker}")
    my = [(t["start"], t["end"]) for t in turns]
    others = [(t["start"] - PAD_S, t["end"] + PAD_S)
              for r in rows if r["speaker"] != speaker for t in r["turns"]]

    tier1 = []
    for a, b in my:
        for c, d in others:
            s, e = max(a, c), min(b, d)
            if e - s >= 0.05:
                tier1.append((s, e))

    report("acoustic overlap scan (segmentation-3.0)", 0.03)
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tier2_all = acoustic_overlaps(job_dir / "audio" / "original.wav", device)
    # keep only acoustic overlaps that touch the target's turns
    tier2 = [(s, e) for s, e in tier2_all
             if any(a < e and b > s for a, b in my)]

    merged = merge_close(sorted(tier1 + tier2), 0.2)
    return my, merged, len(tier1), len(tier2)


# ---------------- elimination engines ----------------

def _separate_window_sam(job_dir, speaker, a, b, model, reranking):
    from .separate import _decode_16k, _remote_with_timeout, speaker_anchors
    import modal
    chunk = job_dir / "_sam_chunk.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}",
         "-i", str(next(job_dir.glob("master.*"),
                        job_dir / "audio" / "original.wav")),
         "-ac", "1", str(chunk)], check=True)
    anchors = speaker_anchors(job_dir, speaker, a, b - a)
    Separator = modal.Cls.from_name("tag-studio-sam-audio", "Separator")
    out = _remote_with_timeout(Separator(), chunk.read_bytes(), "speech",
                               model=model, reranking=reranking, anchors=anchors)
    chunk.unlink(missing_ok=True)
    return _decode_16k(out["target"])


def _spans_audio(x: np.ndarray, spans_rel: list[tuple]) -> np.ndarray | None:
    """Concatenate the given (window-relative) spans of x; None if <0.5s."""
    parts = [x[int(s * 16000):int(e * 16000)] for s, e in spans_rel]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    cat = np.concatenate(parts).astype(np.float32)
    return cat if len(cat) >= 8000 else None


def _separate_window_sepformer(audio16, a, b, centroid, my_spans_rel):
    """Blind 2-speaker separation; assign streams by embedding similarity
    measured ONLY on the target's turn spans (whole-window embeddings are
    silence-polluted in sparse windows)."""
    import torch
    sep = _get_sepformer()
    i0, i1 = int(a * 16000), min(len(audio16), int(b * 16000))
    mix = torch.from_numpy(audio16[i0:i1])[None]
    with torch.inference_mode():
        est = sep.separate_batch(mix)        # (1, T, n_src)
    est = est[0].cpu().numpy()
    est = est / (np.abs(est).max(axis=0, keepdims=True) + 1e-9) \
        * (np.abs(audio16[i0:i1]).max() + 1e-9)
    sims = []
    for k in range(est.shape[1]):
        x = _spans_audio(est[:, k], my_spans_rel)
        if x is None:
            x = est[:, k].astype(np.float32)
        sims.append(float(np.dot(_embed(x), centroid)))
    return est[:, int(np.argmax(sims))].astype(np.float32), max(sims)


# ---------------- purity ----------------

def speaker_centroid(audio16: np.ndarray, my_turns, others) -> np.ndarray:
    """Embedding centroid from up to 8 solo turns of the target speaker."""
    from .intervals import subtract_intervals
    solo = subtract_intervals(my_turns, others)
    solo = [s for s in solo if s[1] - s[0] >= 1.0][:8]
    if not solo:
        solo = sorted(my_turns, key=lambda t: t[0] - t[1])[:4]
    embs = []
    for a, b in solo:
        x = audio16[int(a * 16000):int(b * 16000)]
        if len(x) > 8000:
            embs.append(_embed(x.astype(np.float32)))
    c = np.mean(embs, axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


def purity_of(track: np.ndarray, a: float, b: float, centroid,
              my_spans_rel: list[tuple]) -> float | None:
    """Embed only the target's turn spans within [a,b] — the whole-window
    embedding is silence-polluted when the speaker is sparse. None = too
    little target speech in this window to score meaningfully."""
    win = track[int(a * 16000):int(b * 16000)]
    x = _spans_audio(win, my_spans_rel)
    if x is None:
        return None
    if float(np.sqrt((x ** 2).mean())) < 0.003:
        return 1.0   # silence is pure
    return float(np.dot(_embed(x), centroid))


# ---------------- the lane builder ----------------

def _paste(track: np.ndarray, y: np.ndarray, i0: int, i1: int) -> None:
    """Replace track[i0:i1] with y, 20ms crossfade at both edges (no clicks)."""
    n = min(len(y), i1 - i0)
    xf = min(int(XFADE_S * 16000), n // 2)
    y = y[:n].copy()
    if xf > 0:
        w = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        y[:xf] = track[i0:i0 + xf] * (1 - w) + y[:xf] * w
        y[n - xf:n] = track[i0 + n - xf:i0 + n] * w[::-1] + y[n - xf:n] * (1 - w[::-1])
    track[i0:i0 + n] = y


def build_clean_lane(job_dir: Path, report, speaker: str,
                     model: str = "facebook/sam-audio-base",
                     reranking: int = 1, engine: str = "auto") -> list[str]:
    import soundfile as sf

    from .audio import load_audio, render_masked_track

    audio = load_audio(job_dir / "audio" / "original.wav", 16000)
    total_s = len(audio) / 16000

    my, suspects, n1, n2 = suspect_regions(job_dir, speaker, report)
    rows = [json.loads(l) for l in
            (job_dir / "manifests" / "speakers.jsonl").read_text().splitlines() if l.strip()]
    n_speakers = len(rows)
    others = [(t["start"], t["end"]) for r in rows if r["speaker"] != speaker
              for t in r["turns"]]

    win_idx = sorted({int(s // WINDOW_S) for s, e in suspects}
                     | {int((e - 1e-3) // WINDOW_S) for s, e in suspects})
    windows = [(i * WINDOW_S, min((i + 1) * WINDOW_S, total_s)) for i in win_idx]

    if engine == "auto":
        # measured on telephony 2026-08-07: SAM rerank>=4 beat SepFormer on
        # every window (0.75-0.81 vs 0.64-0.69 purity) — SAM is the default
        engine = "sam"
    if engine == "sam":
        reranking = max(reranking, 4)
    report(f"{len(suspects)} suspect regions (turns:{n1} acoustic:{n2}) -> "
           f"{engine.upper()} on {len(windows)} windows", 0.06)

    # base: masked original — engine-suffixed lane so A/B comparison is easy
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    name = f"sam_{slug}_clean_{engine}"
    dest = job_dir / "audio" / f"{name}.wav"
    render_masked_track(audio, my, dest, 16000)
    track, _ = sf.read(dest, dtype="float32")

    report("computing speaker centroid", 0.08)
    centroid = speaker_centroid(audio, my, others)

    purity = {"speaker": speaker, "engine": engine, "purity_ok": PURITY_OK,
              "windows": []}
    for i, (a, b) in enumerate(windows):
        my_rel = [(max(s, a) - a, min(e, b) - a) for s, e in my
                  if e > a and s < b]
        if engine == "sepformer":
            y, assign_sim = _separate_window_sepformer(audio, a, b, centroid, my_rel)
        else:
            y = _separate_window_sam(job_dir, speaker, a, b, model, reranking)
            assign_sim = None
        i0, i1 = int(a * 16000), min(len(track), int(b * 16000))
        _paste(track, y, i0, i1)
        p = purity_of(track, a, b, centroid, my_rel)
        purity["windows"].append({
            "start": round(a, 1), "end": round(b, 1),
            "purity": None if p is None else round(p, 3),
            "pass": None if p is None else bool(p >= PURITY_OK),
            "sparse": p is None,
            **({"assign_sim": round(assign_sim, 3)} if assign_sim is not None else {}),
        })
        report(f"{engine} window {i + 1}/{len(windows)} ({a:.0f}-{b:.0f}s, "
               f"purity {'n/a (sparse)' if p is None else f'{p:.2f}'})",
               0.08 + 0.88 * (i + 1) / len(windows))

    sf.write(dest, track, 16000, subtype="PCM_16")
    (job_dir / "peaks" / f"{name}.json").unlink(missing_ok=True)
    n_pass = sum(1 for w in purity["windows"] if w["pass"])
    n_scored = sum(1 for w in purity["windows"] if w["pass"] is not None)
    (job_dir / "manifests" / f"clean_purity_{slug}_{engine}.json").write_text(
        json.dumps(purity, indent=1))
    report(f"clean lane ready ({engine}): {n_pass}/{n_scored} scored windows "
           f"pass purity ≥{PURITY_OK} "
           f"({len(windows) - n_scored} too sparse to score)", 1.0)
    return [name]
