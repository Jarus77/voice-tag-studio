"""Beats detectors — [pauses] [hesitates] [stammers].

Pure rules over stage-5 alignment output; no models, near-zero cost.
Recipes from the podcast_pipeline plan:
  pauses    intra-segment silent gap >= 0.5s between aligned words
  hesitates filler token adjacent to a >=0.3s gap, or a 0.3-1.0s
            unaligned island
  stammers  repeated word/part-word with tight timing, or a rapid
            multi-island cluster
"""

from __future__ import annotations

import re
from pathlib import Path

from ..manifests import read_jsonl
from .base import Detector

PAUSE_MIN_S = 0.5
HES_GAP_S = 0.3
ISLAND_HES = (0.3, 1.0)
STAMMER_JOIN_S = 0.3
CLUSTER_N, CLUSTER_WIN_S = 3, 2.0

FILLERS = {"uh", "um", "hmm", "haan", "अ", "अं", "आं", "उह", "हम्म", "हाँ", "अच्छा"}


def _norm(tok: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]", "", tok.lower())


class BeatsRules(Detector):
    name = "beats"
    tags = ["pauses", "hesitates", "stammers"]
    source = "original"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        aligns = {r["seg_id"]: r for r in
                  read_jsonl(job_dir / "manifests" / "alignments.jsonl")}
        if not aligns:
            raise RuntimeError("alignments.jsonl missing — run the align stage first")

        out = []
        for i, seg in enumerate(segments):
            al = aligns.get(seg["seg_id"])
            if not al or not al.get("words"):
                continue
            words = al["words"]
            islands = [tuple(x) for x in al.get("islands", [])]

            # ---- pauses: silent gap between consecutive aligned words.
            # An island in that gap means un-transcribed SOUND, not silence —
            # skip those (they belong to hesitates/reactions).
            for a, b in zip(words, words[1:]):
                gap = b["start"] - a["end"]
                if gap < PAUSE_MIN_S:
                    continue
                mid = (a["end"] + b["start"]) / 2
                if any(s < mid < e for s, e in islands):
                    continue
                out.append({
                    "seg_id": seg["seg_id"], "tag": "pauses",
                    "score": round(min(1.0, gap / 2.0), 3),
                    "detector": "beats_rules",
                    "position_s": round(mid, 2),
                    "evidence": {"gap_s": round(gap, 2),
                                 "after_word": a["w"], "before_word": b["w"]},
                })

            # ---- hesitates: filler token next to a gap, or a short island
            for j, w in enumerate(words):
                if _norm(w["w"]) not in FILLERS:
                    continue
                gap_before = w["start"] - words[j - 1]["end"] if j else 0.0
                gap_after = words[j + 1]["start"] - w["end"] if j + 1 < len(words) else 0.0
                if max(gap_before, gap_after) >= HES_GAP_S:
                    out.append({
                        "seg_id": seg["seg_id"], "tag": "hesitates",
                        "score": round(min(1.0, 0.5 + max(gap_before, gap_after)), 3),
                        "detector": "beats_rules",
                        "position_s": round(w["start"], 2),
                        "evidence": {"filler": w["w"],
                                     "gap_s": round(max(gap_before, gap_after), 2)},
                    })
            for s, e in islands:
                if ISLAND_HES[0] <= e - s <= ISLAND_HES[1]:
                    out.append({
                        "seg_id": seg["seg_id"], "tag": "hesitates",
                        "score": round(min(1.0, (e - s) / ISLAND_HES[1]), 3),
                        "detector": "beats_rules",
                        "position_s": round((s + e) / 2, 2),
                        "evidence": {"island": [round(s, 2), round(e, 2)],
                                     "kind": "unaligned_island"},
                    })

            # ---- stammers: repeated adjacent token with tight timing
            for a, b in zip(words, words[1:]):
                na, nb = _norm(a["w"]), _norm(b["w"])
                if len(na) >= 2 and na == nb and b["start"] - a["end"] <= STAMMER_JOIN_S:
                    out.append({
                        "seg_id": seg["seg_id"], "tag": "stammers",
                        "score": 0.7,
                        "detector": "beats_rules",
                        "position_s": round(a["start"], 2),
                        "evidence": {"repeated": a["w"],
                                     "join_s": round(b["start"] - a["end"], 2)},
                    })
            # rapid multi-island cluster
            if len(islands) >= CLUSTER_N:
                for k in range(len(islands) - CLUSTER_N + 1):
                    win = islands[k:k + CLUSTER_N]
                    if win[-1][1] - win[0][0] <= CLUSTER_WIN_S:
                        out.append({
                            "seg_id": seg["seg_id"], "tag": "stammers",
                            "score": 0.5,
                            "detector": "beats_rules",
                            "position_s": round(win[0][0], 2),
                            "evidence": {"island_cluster": [[round(s, 2), round(e, 2)]
                                                            for s, e in win]},
                        })
                        break
            if (i + 1) % 100 == 0 or i + 1 == len(segments):
                report(f"scanned {i + 1}/{len(segments)}, {len(out)} candidates",
                       (i + 1) / len(segments))
        return out
