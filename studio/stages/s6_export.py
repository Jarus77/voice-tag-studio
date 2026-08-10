"""Stage 6 — Export: the deliverable.

Turns everything the pipeline produced into TTS training rows:

    voice clip  +  "speaker: [tag] verbatim text with [tags] in place"

Tag rendering:
  • utterance-level tags (tones, states, whispers) go at the START of the line
  • positioned tags (laughs, sigh, pauses, hesitates, stammers) go INLINE at
    the word boundary nearest their timestamp — a [pauses] belongs between the
    two words it separates, which is what the model must learn
  • only candidates at/above TAG_THRESHOLDS become tags; every candidate stays
    in the manifest so thresholds can be re-tuned without re-detecting

Every row carries its quality flags (purity, clipped, separated, align status)
so any filtered subset is reproducible — ablate by filtering, not by re-running.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import EXPORT_EXCLUDE, TAG_THRESHOLDS, UTTERANCE_TAGS
from ..manifests import read_json, read_jsonl, write_json, write_jsonl


def _split_of(video_id: str, val: float = 0.05) -> str:
    """Deterministic val split BY VIDEO — never by segment, or the same
    conversation leaks across train/val."""
    h = int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val else "train"


def _render(text: str, words: list[dict], tags: list[dict]) -> str:
    """Insert tag markers into the transcript at their true positions."""
    tokens = text.split()
    lead = [t for t in tags if t["tag"] in UTTERANCE_TAGS or t.get("position_s") is None]
    inline = [t for t in tags if t not in lead]

    # index of the first word starting after the tag's timestamp
    def idx_for(pos: float) -> int:
        for w in words:
            if w["start"] > pos:
                return w["i"]
        return len(tokens)

    at: dict[int, list[str]] = {}
    for t in sorted(inline, key=lambda t: t["position_s"]):
        at.setdefault(idx_for(t["position_s"]), []).append(f"[{t['tag']}]")

    out: list[str] = [f"[{t['tag']}]" for t in
                      sorted(lead, key=lambda t: -t.get("score", 0))]
    for i, tok in enumerate(tokens):
        out.extend(at.get(i, []))
        out.append(tok)
    out.extend(at.get(len(tokens), []))
    return " ".join(out)


def run(job_dir: Path, report) -> None:
    vid = job_dir.name
    m = job_dir / "manifests"
    segments = read_jsonl(m / "segments.jsonl")
    if not segments:
        raise RuntimeError("segments.jsonl missing — run the segment stage first")
    trs = {r["seg_id"]: r for r in read_jsonl(m / "transcripts.jsonl")}
    als = {r["seg_id"]: r for r in read_jsonl(m / "alignments.jsonl")}
    names = read_json(job_dir / "job.json").get("speaker_names") or {}

    # every detector's candidates, thresholded
    cands: dict[str, list[dict]] = {}
    n_seen = n_kept = 0
    for f in sorted(m.glob("candidates_*.jsonl")):
        for c in read_jsonl(f):
            n_seen += 1
            thr = TAG_THRESHOLDS.get(c["tag"])
            if thr is not None and c.get("score", 0) < thr:
                continue
            n_kept += 1
            cands.setdefault(c["seg_id"], []).append(c)

    # silence tags are ALWAYS derived from word timings (silence must be
    # represented in the text — clips may contain bridged silences up to
    # ~2s, and untagged silence is a text/audio mismatch):
    #     gap 0.5–1.5 s  →  [pauses]   (a beat inside the sentence)
    #     gap ≥ 1.5 s    →  [silence]  (a deliberate stop)
    # A detector-provided [pauses] at the same spot is reclassified by the
    # measured gap rather than duplicated.
    PAUSE_MIN_S, SILENCE_MIN_S = 0.5, 1.5
    n_auto = 0
    for al in als.values():
        ws = al.get("words") or []
        for a, b in zip(ws, ws[1:]):
            gap = b["start"] - a["end"]
            if gap < PAUSE_MIN_S:
                continue
            tag = "silence" if gap >= SILENCE_MIN_S else "pauses"
            mid = (a["end"] + b["start"]) / 2
            near = [c for c in cands.get(al["seg_id"], [])
                    if c["tag"] in ("pauses", "silence")
                    and c.get("position_s") is not None
                    and abs(mid - c["position_s"]) < 0.3]
            if near:
                for c in near:
                    c["tag"] = tag
                continue
            cands.setdefault(al["seg_id"], []).append({
                "seg_id": al["seg_id"], "tag": tag,
                "score": round(min(1.0, gap / 2.0), 3),
                "detector": "export_derived", "position_s": round(mid, 2),
                "evidence": {"gap_s": round(gap, 2)}})
            n_auto += 1
    report(f"{n_kept}/{n_seen} candidates pass thresholds · "
           f"{n_auto} [pauses]/[silence] derived from word gaps", 0.3)

    rows, dropped = [], {}
    for seg in segments:
        tr = trs.get(seg["seg_id"], {})
        al = als.get(seg["seg_id"], {})
        text = (tr.get("text") or "").strip()

        flags = {
            "no_text": not text,
            "impure": seg.get("purity") is not None and not seg.get("purity_pass", True),
            "clipped": bool(al.get("clipped")),
            "separated": bool(seg.get("separated")),
            "major_gap": al.get("status") == "major_gap",
            "align_error": al.get("status") == "align_error",
        }
        skip = next((k for k, on in EXPORT_EXCLUDE.items() if on and flags[k]), None)
        if skip:
            dropped[skip] = dropped.get(skip, 0) + 1
            continue

        tags = cands.get(seg["seg_id"], [])
        words = al.get("words") or []
        tagged = _render(text, words, tags) if words else \
            " ".join([f"[{t['tag']}]" for t in tags] + [text])
        speaker = names.get(seg["speaker"], seg["speaker"])

        rows.append({
            "sample_id": seg["seg_id"],
            "video_id": vid,
            "speaker": speaker,
            "speaker_label": seg["speaker"],
            "text": tagged,                 # with [tags] rendered in place
            "text_plain": text,             # verbatim, no tags
            "line": f"{speaker}: {tagged}", # the training string
            "tags": sorted({t["tag"] for t in tags}),
            "wav": seg["wav"],
            "dur": seg["dur"],
            "start": seg["start"],
            "end": seg["end"],
            "split": _split_of(vid),
            # provenance + quality, so any subset is reproducible by filtering
            "source": seg.get("source", "original"),
            "separated_frac": seg.get("separated_frac", 0.0),
            "purity": seg.get("purity"),
            "align_status": al.get("status"),
            "clipped": al.get("clipped") or [],
            "asr": tr.get("asr"),
            "script": tr.get("script"),
        })

    write_jsonl(m / "dataset.jsonl", rows)

    # speaker × tag matrix — the steering artifact: it shows which cells are
    # interpolation at inference and which are extrapolation
    matrix: dict[str, dict[str, int]] = {}
    for r in rows:
        cell = matrix.setdefault(r["speaker"], {"_rows": 0, "_neutral": 0})
        cell["_rows"] += 1
        if not r["tags"]:
            cell["_neutral"] += 1
        for t in r["tags"]:
            cell[t] = cell.get(t, 0) + 1
    write_json(m / "matrix.json", {"video_id": vid, "matrix": matrix,
                                   "dropped": dropped})

    tagged_rows = sum(1 for r in rows if r["tags"])
    dur = sum(r["dur"] for r in rows)
    report(f"{len(rows)} rows ({dur/60:.1f} min) · {tagged_rows} tagged · "
           f"{len(rows)-tagged_rows} neutral"
           + (f" · dropped {dropped}" if dropped else ""), 1.0)
