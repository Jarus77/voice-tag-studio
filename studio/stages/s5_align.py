"""Stage 5 — Word-level forced alignment + coverage islands (MMS recipe).

Ported from the voice pipeline's stage5b: torchaudio MMS_FA (uroman
romanization handles Devanagari + Hinglish Latin) + pyannote segmentation-3.0
VAD. Islands = VAD speech minus (aligned word spans ± WORD_PAD_S), kept if
>= MIN_ISLAND_S. Here islands are not (only) a drop signal — they're
annotation input ([pauses] from gaps, [hesitates]/[stammers] from island
patterns) and, in the UI, the places to LISTEN for untranscribed events.

Fix over the original: VAD Inference() gets an explicit device (the original
silently ran it on CPU).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import (FILLER_MAX_S, MIN_ISLAND_S, MIN_SPEECH_FRAMES, SR,
                      VAD_MODEL, WORD_PAD_S)
from ..intervals import merge_close, subtract_intervals
from ..manifests import read_jsonl, write_jsonl


class Aligner:
    def __init__(self, device: str):
        import torch
        import torchaudio
        import uroman
        from huggingface_hub import get_token
        from pyannote.audio import Inference, Model

        self.torch = torch
        self.device = torch.device(device)
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).to(self.device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.fa = bundle.get_aligner()
        self.ur = uroman.Uroman()
        seg = Model.from_pretrained(VAD_MODEL, use_auth_token=get_token())
        self.vad = Inference(seg, duration=10.0, step=10.0, device=self.device)

    def words(self, text: str) -> tuple[list[str], list[int], int]:
        """Romanize per original token so word spans map back to display text.

        Returns (alignable_romanized, original_token_indices, n_skipped).
        Skipped tokens (digits, ₹...) have real audio but no alignable letters;
        their spans look like islands — callers must mark those uncertain."""
        tokens = text.split()
        out, idx, skipped = [], [], 0
        for i, tok in enumerate(tokens):
            roman = self.ur.romanize_string(tok)
            c = re.sub(r"[^a-z']", "", roman.lower().replace(" ", ""))
            if c:
                out.append(c)
                idx.append(i)
            else:
                skipped += 1
        return out, idx, skipped

    def speech_intervals(self, wav: Path, dur: float) -> list[tuple]:
        out = self.vad(str(wav))
        data = np.asarray(out.data)
        if data.ndim == 2:
            data = data[None]
        n_frames = data.shape[1]
        frame_s = 10.0 / n_frames
        iv = []
        for w, win in enumerate(data):
            act = (win.sum(axis=-1) >= 1).astype(np.int8)
            start = None
            for f, v in enumerate(np.append(act, 0)):
                if v and start is None:
                    start = f
                elif not v and start is not None:
                    if f - start >= MIN_SPEECH_FRAMES:
                        iv.append((w * 10.0 + start * frame_s,
                                   min(w * 10.0 + f * frame_s, dur)))
                    start = None
        return merge_close(iv, 0.10)

    def word_spans(self, wav: Path, words: list[str]) -> list[tuple]:
        import soundfile as sf
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == SR
        x = self.torch.from_numpy(audio)[None].to(self.device)
        with self.torch.inference_mode():
            emission, _ = self.model(x)
        spans = self.fa(emission[0].cpu(), self.tokenizer(words))  # FA on CPU
        ratio = x.size(1) / emission.size(1) / SR
        return [(s[0].start * ratio, s[-1].end * ratio) for s in spans]


def run(job_dir: Path, report) -> None:
    import torch

    segments = read_jsonl(job_dir / "manifests" / "segments.jsonl")
    transcripts = {r["seg_id"]: r for r in
                   read_jsonl(job_dir / "manifests" / "transcripts.jsonl")}
    if not transcripts:
        raise RuntimeError("transcripts.jsonl missing — run asr first")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    report(f"loading MMS_FA + VAD on {device}", 0.02)
    al = Aligner(device)

    rows = []
    todo = [s for s in segments
            if (transcripts.get(s["seg_id"], {}).get("text") or "").strip()]
    for i, seg in enumerate(todo):
        text = transcripts[seg["seg_id"]]["text"].strip()
        tokens = text.split()
        words, tok_idx, skipped = al.words(text)
        row = {"seg_id": seg["seg_id"], "n_words": len(words),
               "n_skipped_words": skipped}
        try:
            if not words:
                raise ValueError("no alignable words")
            wav = job_dir / seg["wav"]
            spans = al.word_spans(wav, words)
            speech = al.speech_intervals(wav, seg["dur"])
            cover = merge_close(
                [(max(0.0, a - WORD_PAD_S), b + WORD_PAD_S) for a, b in spans], 0.05)
            islands = [(a, b) for a, b in subtract_intervals(speech, cover)
                       if b - a >= MIN_ISLAND_S]
            speech_s = sum(b - a for a, b in speech)
            island_s = sum(b - a for a, b in islands)
            mx = max((b - a for a, b in islands), default=0.0)
            if not islands:
                cls = "clean"
            elif skipped > 0:
                cls = "digits_uncertain"
            elif mx <= FILLER_MAX_S:
                cls = "filler_suspect"
            else:
                cls = "major_gap"
            row.update({
                "status": cls,
                "words": [{"w": tokens[ti], "start": round(a, 3), "end": round(b, 3)}
                          for ti, (a, b) in zip(tok_idx, spans)],
                "islands": [[round(a, 2), round(b, 2)] for a, b in islands],
                "aligned_frac": round(1.0 - island_s / speech_s, 3) if speech_s else None,
                "max_island_s": round(mx, 2),
            })
        except Exception as e:
            row.update({"status": "align_error",
                        "error": f"{type(e).__name__}: {str(e)[:120]}"})
        rows.append(row)
        if (i + 1) % 5 == 0 or i + 1 == len(todo):
            report(f"aligned {i + 1}/{len(todo)}", 0.05 + 0.93 * (i + 1) / len(todo))

    write_jsonl(job_dir / "manifests" / "alignments.jsonl", rows)
