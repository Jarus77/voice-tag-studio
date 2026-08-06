"""JSONL manifest I/O + schema notes.

Per-job manifests (jobs/<video_id>/manifests/):

stage0_video.json     {video_id, url, title, channel, duration_s, master_path,
                       master_probe:{...}, wav_path, sample_rate}
speakers.jsonl        {video_id, speaker, total_s, turns:[{start,end}],
                       exclusive_turns:[{start,end}], track_wav, embedding_ok}
segments.jsonl        {seg_id, video_id, speaker, start, end, dur, wav, over_len}
transcripts.jsonl     {seg_id, text, asr, language, script:{...}, error|null}
alignments.jsonl      {seg_id, words:[{w,start,end}], n_skipped_words,
                       islands:[[s,e]], aligned_frac, status}
candidates_<det>.jsonl {seg_id, tag, score, detector, position_s|null, evidence}

All times inside words/islands/position_s are SEGMENT-relative; absolute video
time = seg.start + t (the UI does the addition when seeking full tracks).
"""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)
