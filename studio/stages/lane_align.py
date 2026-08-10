"""Stage — lane align: force-align each chunk's transcript on the lane.

Produces word timestamps on the LANE TIMELINE (absolute seconds), which the
cut stage then uses as the authority for where clips may begin and end.

Output: manifests/lane_alignments.jsonl, one row per chunk:
    {speaker, chunk_id, start, end, tokens: [...],
     words: [{i, w, start, end}]   (ABSOLUTE lane seconds; i = token index),
     islands: [[s, e], ...]        (absolute; audible speech with no text),
     aligned_frac, status, n_skipped_words}
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..config import FILLER_MAX_S, MIN_ISLAND_S, WORD_PAD_S
from ..intervals import merge_close, subtract_intervals
from ..manifests import read_jsonl, write_jsonl
from .lane_asr import CHUNK_PAD_S, lane_wav_for
from .s5_align import Aligner


def run(job_dir: Path, report) -> None:
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
        for i, ch in enumerate(todo):
            lane, _ = lane_wav_for(job_dir, ch["speaker"])
            # exact same slice the ASR heard (incl. the pad)
            off = max(0.0, ch["start"] - CHUNK_PAD_S)
            dur = (ch["end"] - ch["start"]) + 2 * CHUNK_PAD_S
            wav = Path(td) / f"{ch['chunk_id']}.wav"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{off:.3f}",
                 "-t", f"{dur:.3f}", "-i", str(lane),
                 "-c:a", "pcm_s16le", str(wav)], check=True)

            text = ch["text"].strip()
            tokens = text.split()
            words, tok_idx, skipped = al.words(text)
            row = {"speaker": ch["speaker"], "chunk_id": ch["chunk_id"],
                   "start": ch["start"], "end": ch["end"],
                   "tokens": tokens, "n_skipped_words": skipped}
            try:
                if not words:
                    raise ValueError("no alignable words")
                spans = al.word_spans(wav, words)
                speech = al.speech_intervals(wav, dur)
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
            report(f"chunk {i + 1}/{len(todo)} ({ch['chunk_id']}, "
                   f"{row['status']})", 0.05 + 0.93 * (i + 1) / len(todo))

    write_jsonl(job_dir / "manifests" / "lane_alignments.jsonl", rows)
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    report(f"aligned {len(rows)} chunks · " +
           " ".join(f"{k}:{v}" for k, v in c.items()), 1.0)
