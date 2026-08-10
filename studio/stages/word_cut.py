"""Stage — segment (word-aware): cut clips ONLY in verified word gaps.

The inversion's payoff. Words were transcribed and timestamped on the lane
BEFORE cutting, so:
  • every boundary lies in a gap between aligned words (never mid-phrase)
  • each clip's text is DERIVED — the tokens whose spans fall inside it —
    text↔audio match holds by construction
  • short utterances can merge across small bridges into viable clips
  • the old checks (islands, purity, clipped) remain as an acceptance gate

Emits the same manifests downstream stages already consume:
  segments.jsonl · transcripts.jsonl · alignments.jsonl (clip-relative)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import (CLIPPED_EDGE_S, CROSSTALK_GUARD_S, FILLER_MAX_S,
                      MAX_SEG_S, MIN_SEG_S, SR)
from ..intervals import merge_close, subtract_intervals
from ..manifests import read_json, read_jsonl, write_jsonl
from .lane_asr import lane_wav_for
from .s4_asr import script_profile

GROUP_GAP_S = 0.60        # a word gap this long separates utterance groups
MAX_WORD_S = 2.00         # longer single word = alignment glitch, span unusable
BRIDGE_MAX_S = 2.00       # groups may merge across a silence up to this long
BRIDGE_FOREIGN_MAX_S = 0.60   # ...unless the other speaker talks longer than
                              # this inside the bridge (masked silence is fine
                              # for a backchannel, not for a full turn)
PAD_START_S = 0.15
PAD_END_S = 0.20
EDGE_GAP_S = 0.10         # clips never touch a neighbouring group's words
SPLIT_GAP_MIN_S = 0.25    # >20s groups split only at gaps this size or larger
TAIL_ISLAND_EXT_S = 1.0   # trailing digits: extend end over their island


def _forbidden_regions(job_dir: Path, speaker: str, source: str,
                       others: list[tuple], guard: float) -> list[tuple]:
    """Time regions a clip must NOT include.

    masked lane: every other-speaker turn ±guard (overlap = two voices mixed).
    clean lane:  purity-FAILED windows (their separation was reverted to
                 original = mixed audio) AND the rescued regions themselves —
                 SepFormer-reconstructed audio has audible artifacts, so clips
                 cut around it; everything that remains is genuine original
                 audio. Rescue still pays off through ASR/alignment, which DO
                 read the reconstructed seconds."""
    if source == "masked":
        return merge_close([(a - guard, b + guard) for a, b in others], 0.0)
    RESCUE_PAD_S = 0.06          # keep the 20 ms paste crossfades out too
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    for name in (f"clean_purity_{slug}_sepformer.json",
                 f"clean_purity_{slug}_sam.json", f"clean_purity_{slug}.json"):
        p = job_dir / "manifests" / name
        if p.exists():
            d = json.loads(p.read_text())
            bad = [(w["start"], w["end"]) for w in d.get("windows", [])
                   if w.get("pass") is False]
            bad += [(a - RESCUE_PAD_S, b + RESCUE_PAD_S)
                    for a, b in d.get("rescued_regions", [])]
            return merge_close(bad, 0.0)
    return []


def _rescued_regions(job_dir: Path, speaker: str) -> list[tuple]:
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    for name in (f"clean_purity_{slug}_sepformer.json",
                 f"clean_purity_{slug}_sam.json", f"clean_purity_{slug}.json"):
        p = job_dir / "manifests" / name
        if p.exists():
            d = json.loads(p.read_text())
            r = [tuple(x) for x in d.get("rescued_regions", [])]
            return merge_close(r, 0.05)
    return []


def _assign_tokens(tokens: list[str], words: list[dict]) -> list[int | None]:
    """word-index (into `words`) owning each token; unaligned tokens inherit
    their nearest preceding aligned token (or the following one at the top)."""
    own: list[int | None] = [None] * len(tokens)
    for wi, w in enumerate(words):
        own[w["i"]] = wi
    last = None
    for i in range(len(tokens)):
        if own[i] is not None:
            last = own[i]
        else:
            own[i] = last            # may stay None before the first word
    nxt = None
    for i in range(len(tokens) - 1, -1, -1):
        if own[i] is not None:
            nxt = own[i]
        else:
            own[i] = nxt
    return own


def _in_any(t: float, regions: list[tuple]) -> bool:
    return any(a <= t <= b for a, b in regions)


def _overlap_s(a: float, b: float, regions: list[tuple]) -> float:
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in regions)


def run(job_dir: Path, report) -> None:
    vid = job_dir.name
    speakers = read_jsonl(job_dir / "manifests" / "speakers.jsonl")
    chunks = read_jsonl(job_dir / "manifests" / "lane_alignments.jsonl")
    if not chunks:
        raise RuntimeError("lane_alignments.jsonl missing — run align first")
    lane_tr = {c["chunk_id"]: c for c in
               read_jsonl(job_dir / "manifests" / "lane_transcripts.jsonl")}
    cfg = read_json(job_dir / "job.json")
    guard = float(cfg.get("crosstalk_guard") or CROSSTALK_GUARD_S)
    min_seg = float(cfg.get("min_seg") or MIN_SEG_S)
    bridge_max = float(cfg.get("bridge_max") or BRIDGE_MAX_S)
    all_turns = {r["speaker"]: [(t["start"], t["end"]) for t in r["turns"]]
                 for r in speakers}

    seg_rows, tr_rows, al_rows = [], [], []
    dropped_short = {"n": 0, "s": 0.0}
    seg_dir = job_dir / "segments"

    for spk_row in speakers:
        spk = spk_row["speaker"]
        lane, source = lane_wav_for(job_dir, spk)
        others = [iv for o, tv in all_turns.items() if o != spk for iv in tv]
        forbidden = _forbidden_regions(job_dir, spk, source, others, guard)
        rescued = _rescued_regions(job_dir, spk)
        audio = None                     # lazy-load per speaker
        n_spk = 0

        for ch in [c for c in chunks if c["speaker"] == spk]:
            words = ch.get("words") or []
            tokens = ch.get("tokens") or []
            if not words:
                continue
            words = sorted(words, key=lambda w: w["start"])
            own = _assign_tokens(tokens, words)
            islands = [tuple(x) for x in ch.get("islands", [])]

            # a single "word" longer than MAX_WORD_S is a forced-alignment
            # glitch (MMS stretching a token across silence) — its span is
            # unusable audio: treat it like a forbidden region so no clip can
            # contain it or bridge across it. Shrunk 50 ms so the words
            # touching its edges aren't swallowed with it.
            glitch = [(w["start"] + 0.05, w["end"] - 0.05) for w in words
                      if w["end"] - w["start"] > MAX_WORD_S]
            forbid = merge_close(forbidden + glitch, 0.0) if glitch else forbidden

            # words inside forbidden regions can never enter a clip;
            # glitched words are excluded by their own duration (their
            # endpoints sit outside the shrunk glitch span)
            ok = [w["end"] - w["start"] <= MAX_WORD_S
                  and not (_in_any(w["start"], forbid) or _in_any(w["end"], forbid))
                  for w in words]

            # ---- group consecutive usable words at GROUP_GAP boundaries ----
            groups: list[list[int]] = []
            for wi, w in enumerate(words):
                if not ok[wi]:
                    continue
                if (groups and ok[groups[-1][-1]]
                        and w["start"] - words[groups[-1][-1]]["end"] < GROUP_GAP_S
                        and not _overlap_s(words[groups[-1][-1]]["end"],
                                           w["start"], forbid)):
                    groups[-1].append(wi)
                else:
                    groups.append([wi])

            def span(g: list[int]) -> float:
                return words[g[-1]]["end"] - words[g[0]]["start"]

            # ---- merge short groups across small, mostly-quiet bridges ----
            merged: list[list[int]] = []
            for g in groups:
                if (merged and (span(merged[-1]) < min_seg or span(g) < min_seg)):
                    bridge_a = words[merged[-1][-1]]["end"]
                    bridge_b = words[g[0]]["start"]
                    if (bridge_b - bridge_a <= bridge_max
                            and _overlap_s(bridge_a, bridge_b, others) <= BRIDGE_FOREIGN_MAX_S
                            and not _overlap_s(bridge_a, bridge_b, forbid)):
                        merged[-1].extend(g)
                        continue
                merged.append(g)

            # ---- split overlong groups at their largest interior word gap ----
            final: list[list[int]] = []

            def split(g: list[int]) -> None:
                if span(g) <= MAX_SEG_S or len(g) < 2:
                    final.append(g)
                    return
                lo, hi = max(1, len(g) // 5), max(1, len(g) * 4 // 5)
                if hi <= lo:                # tiny group (2-3 words): any gap
                    lo, hi = 1, len(g)
                cand = [(words[g[k]]["start"] - words[g[k - 1]]["end"], k)
                        for k in range(lo, hi)]
                big = [c for c in cand if c[0] >= SPLIT_GAP_MIN_S]
                gap, k = max(big or cand)
                split(g[:k]); split(g[k:])

            for g in merged:
                split(g)

            # ---- emit clips ----
            for g in final:
                # an edge word whose outside neighbour is too close can never
                # get a clean cut (the boundary would land inside a word or
                # within CLIPPED_EDGE of it). Sacrifice that word, keep the
                # clip — better than emitting a `clipped` row the export gate
                # then drops whole.
                while g:
                    prev = words[g[0] - 1]["end"] if g[0] > 0 else -1e12
                    if prev + EDGE_GAP_S <= words[g[0]]["start"] - CLIPPED_EDGE_S:
                        break
                    g = g[1:]
                while g:
                    nxt = words[g[-1] + 1]["start"] if g[-1] + 1 < len(words) else 1e12
                    if words[g[-1]]["end"] + CLIPPED_EDGE_S <= nxt - EDGE_GAP_S:
                        break
                    g = g[:-1]
                if not g:
                    continue
                w0, w1 = words[g[0]], words[g[-1]]
                dur_words = w1["end"] - w0["start"]
                if dur_words < min_seg:
                    dropped_short["n"] += 1
                    dropped_short["s"] += dur_words
                    continue

                # token range: everything owned by this group's words
                gset = set(g)
                tok_ids = [i for i in range(len(tokens)) if own[i] in gset]
                if not tok_ids:
                    continue
                text = " ".join(tokens[i] for i in range(min(tok_ids),
                                                         max(tok_ids) + 1))

                # boundaries: pad into the surrounding gaps, never into a
                # neighbouring word (words are time-sorted, so index
                # neighbours ARE the time neighbours)
                prev_end = words[g[0] - 1]["end"] if g[0] > 0 else 0.0
                nxt_start = words[g[-1] + 1]["start"] if g[-1] + 1 < len(words) else 1e12
                ts = max(w0["start"] - PAD_START_S, prev_end + EDGE_GAP_S, 0.0)
                te = min(w1["end"] + PAD_END_S, nxt_start - EDGE_GAP_S)
                # trailing unaligned tokens (digits): extend over their island
                if any(i > w1["i"] for i in tok_ids):
                    ext = [b for a, b in islands
                           if w1["end"] - 0.2 <= a <= w1["end"] + TAIL_ISLAND_EXT_S]
                    if ext:
                        te = min(max(te, ext[0] + 0.1), nxt_start - EDGE_GAP_S)
                if te - ts < min_seg or te <= ts:
                    dropped_short["n"] += 1
                    dropped_short["s"] += max(0.0, te - ts)
                    continue

                if audio is None:
                    audio, _sr = sf.read(lane, dtype="float32")
                seg_id = f"{vid}_{spk}_{n_spk:04d}"
                dest = seg_dir / spk / f"{seg_id}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                i0, i1 = int(ts * SR), min(len(audio), int(te * SR))
                sf.write(dest, audio[i0:i1], SR, subtype="PCM_16")

                sep_s = _overlap_s(ts, te, rescued)
                clip_words = [{"i": words[wi]["i"] - min(tok_ids),
                               "w": words[wi]["w"],
                               "start": round(words[wi]["start"] - ts, 3),
                               "end": round(words[wi]["end"] - ts, 3)}
                              for wi in g]
                clip_islands = [[round(max(a, ts) - ts, 2), round(min(b, te) - ts, 2)]
                                for a, b in islands if b > ts and a < te]
                isl_s = sum(b - a for a, b in clip_islands)
                n_unaligned = (max(tok_ids) - min(tok_ids) + 1) - len(clip_words)
                mx = max((b - a for a, b in clip_islands), default=0.0)
                status = ("clean" if not clip_islands
                          else "digits_uncertain" if n_unaligned > 0
                          else "filler_suspect" if mx <= FILLER_MAX_S
                          else "major_gap")
                clipped = []
                if clip_words:
                    if clip_words[0]["start"] <= CLIPPED_EDGE_S:
                        clipped.append("start")
                    if clip_words[-1]["end"] >= (te - ts) - CLIPPED_EDGE_S:
                        clipped.append("end")

                seg_rows.append({
                    "seg_id": seg_id, "video_id": vid, "speaker": spk,
                    "start": round(ts, 3), "end": round(te, 3),
                    "dur": round(te - ts, 3),
                    "wav": f"segments/{spk}/{seg_id}.wav",
                    "over_len": bool(te - ts > MAX_SEG_S),
                    "source": source,
                    "separated": bool(sep_s > 0.05),
                    "separated_s": round(sep_s, 2),
                    "separated_frac": round(sep_s / (te - ts), 3),
                })
                tr_rows.append({
                    "seg_id": seg_id, "text": text,
                    "asr": lane_tr.get(ch["chunk_id"], {}).get("engine", "lane"),
                    "language": "",
                    "script": script_profile(text) if text else None,
                    "error": None,
                })
                cover = sum(w["end"] - w["start"] for w in clip_words)
                al_rows.append({
                    "seg_id": seg_id, "n_words": len(clip_words),
                    "n_tokens": max(tok_ids) - min(tok_ids) + 1,
                    "n_skipped_words": n_unaligned,
                    "status": status, "words": clip_words,
                    "islands": clip_islands,
                    "aligned_frac": round(cover / (cover + isl_s), 3)
                                    if (cover + isl_s) else None,
                    "max_island_s": round(mx, 2),
                    "clipped": clipped,
                })
                n_spk += 1
        report(f"{spk}: {n_spk} clips from {source}",
               0.1 + 0.8 * (speakers.index(spk_row) + 1) / len(speakers))

    # ---- purity gate (unchanged: record, don't enforce) ----
    try:
        from ..audio import load_audio
        from ..clean_lane import PURITY_OK, _embed, speaker_centroid
        original = load_audio(job_dir / "audio" / "original.wav", SR)
        report("purity gate", 0.92)
        cents = {}
        for spk_row in speakers:
            spk = spk_row["speaker"]
            my = [(t["start"], t["end"]) for t in spk_row["turns"]]
            oth = [iv for o, tv in all_turns.items() if o != spk for iv in tv]
            try:
                cents[spk] = speaker_centroid(original, my, oth)
            except Exception:
                cents[spk] = None
        for i, r in enumerate(seg_rows):
            c = cents.get(r["speaker"])
            if c is None:
                continue
            try:
                x, _ = sf.read(job_dir / r["wav"], dtype="float32")
                p = float(np.dot(_embed(x[:160000]), c))
                r["purity"] = round(p, 3)
                r["purity_pass"] = bool(p >= PURITY_OK)
            except Exception:
                pass
    except Exception as e:
        report(f"purity gate skipped: {type(e).__name__}", 0.95)

    write_jsonl(job_dir / "manifests" / "segments.jsonl", seg_rows)
    write_jsonl(job_dir / "manifests" / "transcripts.jsonl", tr_rows)
    write_jsonl(job_dir / "manifests" / "alignments.jsonl", al_rows)
    kept = sum(r["dur"] for r in seg_rows)
    n_clean = sum(1 for r in al_rows if r["status"] == "clean")
    report(f"{len(seg_rows)} clips ({kept / 60:.1f} min) · {n_clean} clean · "
           f"dropped {dropped_short['n']} shorts ({dropped_short['s']:.1f}s)", 1.0)
