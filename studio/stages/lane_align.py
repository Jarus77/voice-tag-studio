"""Stage — lane align: force-align each chunk's transcript on the lane.

Produces word timestamps on the LANE TIMELINE (absolute seconds), which the
cut stage then uses as the authority for where clips may begin and end.

The heavy half (wav2vec2 emissions + span decode) runs on the deployed
`voice-tag-align` Modal GPU app when available — chunks are spawned in batches
across containers — and falls back to local MPS/CPU otherwise
(VTS_ALIGN_BACKEND=auto|modal|local). Romanization, VAD and island logic are
always local.

Output: manifests/lane_alignments.jsonl, one row per chunk:
    {speaker, chunk_id, start, end, tokens: [...],
     words: [{i, w, start, end}]   (ABSOLUTE lane seconds; i = token index),
     islands: [[s, e], ...]        (absolute; audible speech with no text),
     aligned_frac, status, n_skipped_words}
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from ..config import FILLER_MAX_S, MIN_ISLAND_S, WORD_PAD_S
from ..intervals import merge_close, subtract_intervals
from ..manifests import read_jsonl, write_jsonl
from .lane_asr import CHUNK_PAD_S, lane_wav_for
from .s5_align import Aligner


def _modal_spans(todo, wavs, prep, report) -> dict:
    """Word spans for every alignable chunk via the Modal GPU app.
    Returns {chunk_id: [[a, b], ...] | {"error": ...}}. Raises on app-level
    failure so the caller can fall back to local."""
    import io

    import modal
    import numpy as np
    import soundfile as sf
    A = modal.Cls.from_name("voice-tag-align", "Aligner")()

    items = []                                # (todo_idx, audio, words)
    for i, (words, _idx, _skip) in enumerate(prep):
        if not words:
            continue
        audio, _sr = sf.read(wavs[i], dtype="float32")
        items.append((i, audio, words))

    B = 8
    payloads: list[tuple[bytes, list[int]]] = []
    for k in range(0, len(items), B):
        grp = items[k:k + B]
        T = max(len(a) for _, a, _ in grp)
        arr = np.zeros((len(grp), T), dtype=np.float16)
        for j, (_, a, _) in enumerate(grp):
            arr[j, :len(a)] = a
        buf = io.BytesIO()
        np.savez_compressed(
            buf, audio=arr,
            lens=np.array([len(a) for _, a, _ in grp], dtype=np.int64),
            words=np.frombuffer(
                json.dumps([w for _, _, w in grp]).encode(), dtype=np.uint8))
        payloads.append((buf.getvalue(), [i for i, _, _ in grp]))

    handles = [A.align.spawn(blob) for blob, _ in payloads]
    out, done = {}, 0
    for h, (blob, idxs) in zip(handles, payloads):
        try:
            res = json.loads(h.get(timeout=600))
        except Exception:                     # one respawn covers wedges
            res = json.loads(A.align.spawn(blob).get(timeout=600))
        for i, sp in zip(idxs, res):
            out[todo[i]["chunk_id"]] = sp
            done += 1
        report(f"mms_fa [gpu:modal] {done}/{len(items)} chunks",
               0.10 + 0.50 * done / len(items))
    return out


def run(job_dir: Path, report) -> None:
    import os

    import torch

    chunks = read_jsonl(job_dir / "manifests" / "lane_transcripts.jsonl")
    if not chunks:
        raise RuntimeError("lane_transcripts.jsonl missing — run asr first")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    report(f"loading MMS_FA + VAD on {device}", 0.02)
    al = Aligner(device)

    todo = [c for c in chunks if (c.get("text") or "").strip()]
    rows = []
    with tempfile.TemporaryDirectory(prefix="lane_align_", dir=job_dir) as td:
        # cut every chunk slice up front (parallel ffmpeg), same slice the
        # ASR heard (incl. the pad)
        from concurrent.futures import ThreadPoolExecutor

        def cut(ch):
            lane, _ = lane_wav_for(job_dir, ch["speaker"])
            off = max(0.0, ch["start"] - CHUNK_PAD_S)
            dur = (ch["end"] - ch["start"]) + 2 * CHUNK_PAD_S
            wav = Path(td) / f"{ch['chunk_id']}.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{off:.3f}",
                 "-t", f"{dur:.3f}", "-i", str(lane),
                 "-c:a", "pcm_s16le", str(wav)], check=True)
            return wav

        report(f"cutting {len(todo)} chunk slices", 0.04)
        with ThreadPoolExecutor(max_workers=8) as pool:
            wavs = list(pool.map(cut, todo))
        prep = [al.words(ch["text"].strip()) for ch in todo]

        # word spans: Modal GPU first, local fallback
        spans_by: dict = {}
        backend = f"local:{device}"
        mode = os.environ.get("VTS_ALIGN_BACKEND", "auto")
        if mode in ("auto", "modal"):
            try:
                spans_by = _modal_spans(todo, wavs, prep, report)
                backend = "gpu:modal"
            except Exception as e:
                if mode == "modal":
                    raise
                report(f"modal unavailable ({type(e).__name__}) — "
                       f"local {device}", 0.10)

        # VAD once per LANE, not per chunk: one sliding-window pass over the
        # full lane (pyannote batches it) replaces a per-chunk Inference call
        # with its per-call overhead — this was the stage's residual bottleneck
        import soundfile as sf
        speech_by: dict[str, list] = {}
        spks = sorted({c["speaker"] for c in todo})
        for si, spk in enumerate(spks):
            lane, _ = lane_wav_for(job_dir, spk)
            report(f"vad pass {si + 1}/{len(spks)} ({spk} lane)",
                   0.62 + 0.12 * si / len(spks))
            speech_by[spk] = al.speech_intervals(
                lane, sf.info(str(lane)).duration)

        for i, ch in enumerate(todo):
            off = max(0.0, ch["start"] - CHUNK_PAD_S)
            dur = (ch["end"] - ch["start"]) + 2 * CHUNK_PAD_S
            text = ch["text"].strip()
            tokens = text.split()
            words, tok_idx, skipped = prep[i]
            row = {"speaker": ch["speaker"], "chunk_id": ch["chunk_id"],
                   "start": ch["start"], "end": ch["end"],
                   "tokens": tokens, "n_skipped_words": skipped}
            try:
                if not words:
                    raise ValueError("no alignable words")
                got = spans_by.get(ch["chunk_id"])
                if isinstance(got, dict):
                    raise ValueError(got.get("error", "gpu align failed"))
                spans = ([tuple(s) for s in got] if got is not None
                         else al.word_spans(wavs[i], words))
                end_abs = off + dur
                speech = [(max(a, off) - off, min(b, end_abs) - off)
                          for a, b in speech_by[ch["speaker"]]
                          if b > off and a < end_abs]
                cover = merge_close([(max(0.0, a - WORD_PAD_S), b + WORD_PAD_S)
                                     for a, b in spans], 0.05)
                islands = [(a, b) for a, b in subtract_intervals(speech, cover)
                           if b - a >= MIN_ISLAND_S]
                speech_s = sum(b - a for a, b in speech)
                island_s = sum(b - a for a, b in islands)
                mx = max((b - a for a, b in islands), default=0.0)
                status = ("clean" if not islands
                          else "digits_uncertain" if skipped > 0
                          else "filler_suspect" if mx <= FILLER_MAX_S
                          else "major_gap")
                row.update({
                    "status": status,
                    "words": [{"i": ti, "w": tokens[ti],
                               "start": round(off + a, 3),
                               "end": round(off + b, 3)}
                              for ti, (a, b) in zip(tok_idx, spans)],
                    "islands": [[round(off + a, 2), round(off + b, 2)]
                                for a, b in islands],
                    "aligned_frac": round(1.0 - island_s / speech_s, 3)
                                    if speech_s else None,
                })
            except Exception as e:
                row.update({"status": "align_error", "words": [], "islands": [],
                            "error": f"{type(e).__name__}: {str(e)[:120]}"})
            rows.append(row)
            report(f"islands {i + 1}/{len(todo)} ({ch['chunk_id']}, "
                   f"{row['status']})", 0.76 + 0.22 * (i + 1) / len(todo))

    write_jsonl(job_dir / "manifests" / "lane_alignments.jsonl", rows)
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    report(f"aligned {len(rows)} chunks [{backend}] · " +
           " ".join(f"{k}:{v}" for k, v in c.items()), 1.0)
