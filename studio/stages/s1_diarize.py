"""Stage 1 — Diarize: pyannote community-1 -> speakers.jsonl + masked tracks.

pyannote 4.x apply() returns DiarizeOutput:
  .speaker_diarization            all turns (may overlap)     -> listening tracks
  .exclusive_speaker_diarization  overlap-free turns          -> ASR segment cutting
  .speaker_embeddings             (n_speakers, dim), sorted by labels()

Known gotcha carried over from the voice pipeline: pyannote zero-pads the
embeddings array when there are more speakers than centroids — drop all-zero /
non-finite rows explicitly or junk vectors survive into any later clustering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import load_audio, render_masked_track
from ..config import DIARIZATION_MODEL, MAX_SPEAKERS, SR
from ..manifests import write_jsonl


def _turns_of(annotation) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for seg, _, label in annotation.itertracks(yield_label=True):
        out.setdefault(str(label), []).append((round(seg.start, 3), round(seg.end, 3)))
    return out


def run(job_dir: Path, report) -> None:
    import torch
    from pyannote.audio import Pipeline

    vid = job_dir.name
    wav = job_dir / "audio" / "original.wav"
    if not wav.exists():
        raise RuntimeError("audio/original.wav missing — run ingest first")

    from ..manifests import read_json
    model_id = read_json(job_dir / "job.json").get("diarizer") or DIARIZATION_MODEL
    short = model_id.split("/")[-1]
    report(f"loading {short}", 0.02)
    pipe = Pipeline.from_pretrained(model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe.to(torch.device(device))
    for attr in ("segmentation_batch_size", "embedding_batch_size"):
        try:
            setattr(pipe, attr, 32)   # 4.x defaults both to 1 — far too slow
        except Exception:
            pass

    report(f"{short} on {device}", 0.05)

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        if total:
            report(f"{short}/{step_name} {completed}/{total}",
                   0.05 + 0.55 * (completed / total))

    out = pipe(str(wav), min_speakers=1, max_speakers=MAX_SPEAKERS, hook=hook)

    diar = getattr(out, "speaker_diarization", out)
    excl = getattr(out, "exclusive_speaker_diarization", diar)
    embeddings = getattr(out, "speaker_embeddings", None)
    labels = [str(l) for l in diar.labels()]

    turns = _turns_of(diar)
    excl_turns = _turns_of(excl)

    # drop zero-padded / non-finite embedding rows (pyannote gotcha)
    emb_ok: dict[str, bool] = {}
    kept_rows, kept_labels = [], []
    if embeddings is not None:
        embeddings = np.asarray(embeddings)
        for i, label in enumerate(labels):
            row = embeddings[i] if i < len(embeddings) else None
            ok = (row is not None and np.isfinite(row).all()
                  and float(np.abs(row).sum()) > 0)
            emb_ok[label] = bool(ok)
            if ok:
                kept_rows.append(row)
                kept_labels.append(label)
    if kept_rows:
        np.save(job_dir / "manifests" / "embeddings.npy", np.stack(kept_rows))
        (job_dir / "manifests" / "embeddings_labels.json").write_text(
            __import__("json").dumps(kept_labels))

    report("rendering per-speaker masked tracks", 0.65)
    audio = load_audio(wav, SR)
    rows = []
    spk_dir = job_dir / "audio" / "speakers"
    for i, label in enumerate(sorted(turns)):
        track = spk_dir / f"{label}.wav"
        render_masked_track(audio, turns[label], track, SR)
        rows.append({
            "video_id": vid,
            "speaker": label,
            "total_s": round(sum(e - s for s, e in turns[label]), 2),
            "turns": [{"start": s, "end": e} for s, e in turns[label]],
            "exclusive_turns": [{"start": s, "end": e}
                                for s, e in excl_turns.get(label, [])],
            "track_wav": f"audio/speakers/{label}.wav",
            "embedding_ok": emb_ok.get(label, False),
        })
        report(f"masked track {label}", 0.65 + 0.3 * (i + 1) / max(1, len(turns)))

    write_jsonl(job_dir / "manifests" / "speakers.jsonl", rows)
