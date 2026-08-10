"""Overlap-rescue v2: clean per-speaker lanes with a measurable quality gate.

    detection   tier 1: sweep over pyannote turns (+0.4s dilation)
                tier 2: acoustic — segmentation-3.0 frame-level "≥2 speakers"
                (catches backchannels the diarizer never emitted as turns)
    eliminate   SepFormer (local, discriminative, free) + embedding-based
                stream assignment
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

from .intervals import merge_close, subtract_intervals

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

def _spans_audio(x: np.ndarray, spans_rel: list[tuple]) -> np.ndarray | None:
    """Concatenate the given (window-relative) spans of x; None if <0.5s.
    Capped at 10s total — enough for an embedding, bounded CPU."""
    parts = [x[int(s * 16000):int(e * 16000)] for s, e in spans_rel]
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    cat = np.concatenate(parts).astype(np.float32)[:160000]
    return cat if len(cat) >= 8000 else None


def _pick_stream(est, mix_peak, my_spans_rel, centroid):
    """Of the two separated streams, keep the one matching the target's
    voiceprint — similarity measured ONLY on the target's turn spans
    (whole-window embeddings are silence-polluted in sparse windows)."""
    est = est / (np.abs(est).max(axis=0, keepdims=True) + 1e-9) \
        * (mix_peak + 1e-9)
    sims = []
    for k in range(est.shape[1]):
        x = _spans_audio(est[:, k], my_spans_rel)
        if x is None:
            x = est[:, k].astype(np.float32)
        sims.append(float(np.dot(_embed(x), centroid)))
    return est[:, int(np.argmax(sims))].astype(np.float32), max(sims)


def _sep_local(mixes: list[np.ndarray], progress) -> list[np.ndarray]:
    """CPU separation, 4 windows per forward pass — torch spreads the batch
    across all cores, so this IS the CPU parallelization."""
    import torch
    sep = _get_sepformer()
    out: list[np.ndarray] = []
    B = 4
    for i in range(0, len(mixes), B):
        chunk = mixes[i:i + B]
        T = max(len(m) for m in chunk)
        mb = torch.zeros(len(chunk), T)
        for j, m in enumerate(chunk):
            mb[j, :len(m)] = torch.from_numpy(m)
        with torch.inference_mode():
            est = sep.separate_batch(mb).cpu().numpy()    # (B, T, 2)
        for j, m in enumerate(chunk):
            out.append(est[j, :len(m)].astype(np.float32))
        progress(len(out))
    return out


def _sep_modal(mixes: list[np.ndarray], progress) -> list[np.ndarray]:
    """GPU separation on the deployed `voice-tag-sepformer` Modal app.
    All chunks are spawned up front (Modal fans them out over containers),
    then gathered in order; one respawn per chunk covers transient wedges."""
    import io

    import modal
    Sep = modal.Cls.from_name("voice-tag-sepformer", "Separator")()
    B = 8
    chunks: list[tuple[bytes, list[int]]] = []
    for i in range(0, len(mixes), B):
        group = mixes[i:i + B]
        T = max(len(m) for m in group)
        arr = np.zeros((len(group), T), dtype=np.float16)
        for j, m in enumerate(group):
            arr[j, :len(m)] = m
        buf = io.BytesIO()
        np.savez_compressed(buf, mixes=arr)
        chunks.append((buf.getvalue(), [len(m) for m in group]))
    handles = [Sep.separate.spawn(blob) for blob, _ in chunks]
    out: list[np.ndarray] = []
    for h, (blob, lens) in zip(handles, chunks):
        try:
            got = h.get(timeout=600)
        except Exception:
            got = Sep.separate.spawn(blob).get(timeout=600)
        est = np.load(io.BytesIO(got))["est"].astype(np.float32)
        for j, ln in enumerate(lens):
            out.append(est[j, :ln])
        progress(len(out))
    return out


def _separate_all(mixes: list[np.ndarray], report, frac):
    """Backend chooser. VTS_SEPFORMER_BACKEND=auto|modal|cpu (default auto:
    try the Modal GPU app, fall back to batched CPU)."""
    import os
    mode = os.environ.get("VTS_SEPFORMER_BACKEND", "auto")
    if mode in ("auto", "modal"):
        try:
            out = _sep_modal(mixes, lambda d: report(
                f"sepformer [gpu:modal] {d}/{len(mixes)} windows", frac(d)))
            return out, "gpu:modal"
        except Exception as e:
            if mode == "modal":
                raise
            report(f"modal unavailable ({type(e).__name__}) — "
                   f"batched CPU fallback", frac(0))
    out = _sep_local(mixes, lambda d: report(
        f"sepformer [cpu, batch 4] {d}/{len(mixes)} windows", frac(d)))
    return out, "cpu"


# ---------------- purity ----------------

def speaker_centroid(audio16: np.ndarray, my_turns, others) -> np.ndarray:
    """Embedding centroid from up to 8 solo turns of the target speaker.

    Each snippet is CAPPED at 5s — ecapa needs a few seconds of voice, and
    embedding a multi-minute solo stretch on CPU is what once hung a build
    for 13 minutes."""
    solo = subtract_intervals(my_turns, others)
    solo = [s for s in solo if s[1] - s[0] >= 1.0][:8]
    if not solo:
        solo = sorted(my_turns, key=lambda t: t[0] - t[1])[:4]
    embs = []
    for a, b in solo:
        b = min(b, a + 5.0)
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
                     engine: str = "sepformer") -> list[str]:
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

    # SepFormer only (user, 2026-08-07): discriminative masking beat SAM's
    # generative re-synthesis by ear, and it runs locally for free.
    engine = "sepformer"
    report(f"{len(suspects)} suspect regions (turns:{n1} acoustic:{n2}) -> "
           f"{engine.upper()} on {len(windows)} windows", 0.06)

    # base: masked original — engine-suffixed lane so A/B comparison is easy
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    name = f"sam_{slug}_clean_{engine}"
    dest = job_dir / "audio" / f"{name}.wav"
    # build into a temp file and rename at the end: a lane must never be
    # readable in a half-built state (an interrupted build previously left a
    # masked-original lane that looked finished to the segment stage)
    tmp = job_dir / "audio" / f".{name}.building.wav"
    render_masked_track(audio, my, tmp, 16000)
    track, _ = sf.read(tmp, dtype="float32")

    report("computing speaker centroid", 0.08)
    centroid = speaker_centroid(audio, my, others)

    # ---- separation first: batched (GPU when the Modal app is deployed) and
    # cached per window. SepFormer's output depends only on the window audio,
    # not on which speaker we're building — so every speaker's build shares
    # the cache and the 2nd/3rd lane separates (almost) nothing.
    cache_dir = job_dir / "manifests" / "sep_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ests: dict[int, np.ndarray] = {}
    todo: list[int] = []
    mixes: list[np.ndarray] = []
    for wi, (a, b) in enumerate(windows):
        cf = cache_dir / f"{a:.2f}_{b:.2f}.npy"
        if cf.exists():
            ests[wi] = np.load(cf).astype(np.float32)
        else:
            todo.append(wi)
            mixes.append(audio[int(a * 16000):min(len(audio), int(b * 16000))]
                         .astype(np.float32))
    if len(windows) > len(todo):
        report(f"{len(windows) - len(todo)}/{len(windows)} windows already "
               f"separated (cache)", 0.09)
    if todo:
        outs, backend = _separate_all(
            mixes, report, lambda d: 0.10 + 0.65 * d / len(mixes))
        for wi, est in zip(todo, outs):
            a, b = windows[wi]
            np.save(cache_dir / f"{a:.2f}_{b:.2f}.npy", est.astype(np.float16))
            ests[wi] = est

    purity = {"speaker": speaker, "engine": engine, "purity_ok": PURITY_OK,
              "windows": []}
    rescued_regions: list[tuple] = []
    for i, (a, b) in enumerate(windows):
        my_rel = [(max(s, a) - a, min(e, b) - a) for s, e in my
                  if e > a and s < b]
        i0m, i1m = int(a * 16000), min(len(audio), int(b * 16000))
        y, assign_sim = _pick_stream(ests[i], np.abs(audio[i0m:i1m]).max(),
                                     my_rel, centroid)

        # Paste separated audio ONLY over the real overlap seconds that are
        # also this speaker's speech. Solo speech inside the window keeps the
        # ORIGINAL (higher fidelity than any separation) — separation is a
        # repair, not a replacement.
        paste = []
        for s, e in suspects:
            s2, e2 = max(s, a), min(e, b)
            if e2 - s2 <= 0:
                continue
            for ms, me in my:
                ps, pe = max(s2, ms), min(e2, me)
                if pe - ps > 0.02:
                    paste.append((max(a, ps - 0.1), min(b, pe + 0.1)))
        paste = merge_close(paste, 0.05)

        i0w, i1w = int(a * 16000), min(len(track), int(b * 16000))
        before = track[i0w:i1w].copy()
        for ps, pe in paste:
            j0, j1 = int((ps - a) * 16000), int((pe - a) * 16000)
            k0, k1 = int(ps * 16000), min(len(track), int(pe * 16000))
            if k1 > k0 and j1 > j0:
                _paste(track, y[j0:j1], k0, k1)

        p = purity_of(track, a, b, centroid, my_rel)
        passed = p is not None and p >= PURITY_OK
        if p is not None and not passed:
            track[i0w:i1w] = before      # failed the gate -> keep the original
        elif passed:
            rescued_regions.extend(paste)

        purity["windows"].append({
            "start": round(a, 1), "end": round(b, 1),
            "purity": None if p is None else round(p, 3),
            "pass": None if p is None else bool(passed),
            "sparse": p is None,
            "rescued_s": round(sum(e - s for s, e in paste), 2) if passed else 0.0,
            **({"assign_sim": round(assign_sim, 3)} if assign_sim is not None else {}),
        })
        report(f"{engine} paste+gate {i + 1}/{len(windows)} ({a:.0f}-{b:.0f}s, "
               f"purity {'n/a (sparse)' if p is None else f'{p:.2f}'}"
               f"{'' if passed else ' — reverted to original'})",
               0.76 + 0.19 * (i + 1) / len(windows))
    purity["rescued_regions"] = [[round(s, 2), round(e, 2)]
                                 for s, e in merge_close(rescued_regions, 0.05)]

    sf.write(tmp, track, 16000, subtype="PCM_16")
    tmp.replace(dest)                      # atomic: now the lane exists
    for stale in (job_dir / "peaks").glob(f"{name}.*.json"):
        stale.unlink(missing_ok=True)
    n_pass = sum(1 for w in purity["windows"] if w["pass"])
    n_scored = sum(1 for w in purity["windows"] if w["pass"] is not None)
    resc_s = sum(e - s for s, e in merge_close(rescued_regions, 0.05))
    (job_dir / "manifests" / f"clean_purity_{slug}_{engine}.json").write_text(
        json.dumps(purity, indent=1))
    report(f"clean lane ready: {resc_s:.1f}s of overlap rescued "
           f"({n_pass}/{n_scored} windows passed purity ≥{PURITY_OK}; "
           f"the rest kept the original)", 1.0)
    return [name]
