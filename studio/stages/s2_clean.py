"""Stage 2 — Clean: SepFormer overlap rescue for every diarized speaker.

Builds one clean lane per speaker: the ORIGINAL audio wherever they speak
alone, SepFormer-separated audio only at the overlap seconds, silence
elsewhere. Windows whose separation fails the ECAPA purity gate revert to the
original, so a failed rescue costs nothing.

Segment then cuts from these lanes and KEEPS the rescued overlap instead of
discarding it. Runs locally and free — no API, no GPU rental.
"""

from __future__ import annotations

from pathlib import Path

from ..clean_lane import build_clean_lane
from ..manifests import read_jsonl, write_json


def run(job_dir: Path, report) -> None:
    speakers = read_jsonl(job_dir / "manifests" / "speakers.jsonl")
    if not speakers:
        raise RuntimeError("speakers.jsonl missing — run diarize first")

    built, failed = {}, {}
    for i, row in enumerate(speakers):
        spk = row["speaker"]
        prefix = f"[{i + 1}/{len(speakers)}] {spk}"

        def sub(msg: str, progress: float | None = None, _p=prefix, _i=i) -> None:
            report(f"{_p}: {msg}",
                   (_i + (progress or 0)) / len(speakers))

        try:
            names = build_clean_lane(job_dir, sub, spk)
            built[spk] = names[0] if names else None
        except Exception as e:
            # one speaker failing must not block the rest (a speaker with
            # almost no speech can't produce a voiceprint, for instance)
            failed[spk] = f"{type(e).__name__}: {str(e)[:120]}"
            report(f"{prefix}: skipped — {failed[spk]}", (i + 1) / len(speakers))

    write_json(job_dir / "manifests" / "clean_lanes.json",
               {"built": built, "failed": failed})
    msg = f"{len(built)}/{len(speakers)} speakers cleaned"
    if failed:
        msg += f" · {len(failed)} skipped"
    report(msg, 1.0)
