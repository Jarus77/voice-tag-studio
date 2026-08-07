"""Stage 3 — Segment: per-speaker exclusive turns -> 2-20s training utterances.

Ported segmentation discipline from the voice pipeline: subtract other
speakers' turns ± guard (crosstalk poison), merge small gaps, split long turns
at the quietest interior point, energy-trim edges. Cut from the ORIGINAL 16k
working wav (dataset master policy).

Tag protection: if any candidates_<detector>.jsonl exists, detected event
positions ±NO_CUT_ZONE_PAD_S become no-cut zones so a rerun never bisects a
[laughs]/[sigh] (voice -> event -> same-voice must stay in one segment).
Intervals that cannot be split without cutting an event are kept over-length
and flagged over_len.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from ..audio import load_audio
from ..config import (CROSSTALK_GUARD_S, MAX_SEG_S, MERGE_GAP_S, MIN_SEG_S,
                      NO_CUT_ZONE_PAD_S, SR)
from ..intervals import energy_trim, merge_close, split_long, subtract_intervals
from ..manifests import read_jsonl, write_jsonl


def _clean_lane_for(job_dir: Path, speaker: str) -> tuple:
    """(wav_path, engine, purity_passing_windows) for a speaker's clean lane.

    SepFormer preferred (user's locked choice), SAM next, legacy last.
    Returns (None, None, []) when no clean lane was built for this speaker."""
    import json
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    for engine in ("sepformer", "sam", None):
        name = f"sam_{slug}_clean_{engine}" if engine else f"sam_{slug}_clean"
        wav = job_dir / "audio" / f"{name}.wav"
        if not wav.exists():
            continue
        passing = []
        pf = job_dir / "manifests" / (
            f"clean_purity_{slug}_{engine}.json" if engine
            else f"clean_purity_{slug}.json")
        if pf.exists():
            for w in json.loads(pf.read_text()).get("windows", []):
                if w.get("pass"):
                    passing.append((w["start"], w["end"]))
        return wav, (engine or "legacy"), merge_close(passing, 0.0)
    return None, None, []


def _no_cut_zones(job_dir: Path) -> list[tuple]:
    """Absolute-time protection windows from any prior detector run."""
    seg_start = {r["seg_id"]: r["start"]
                 for r in read_jsonl(job_dir / "manifests" / "segments.jsonl")}
    zones = []
    for cand_file in (job_dir / "manifests").glob("candidates_*.jsonl"):
        for c in read_jsonl(cand_file):
            pos = c.get("position_s")
            base = seg_start.get(c.get("seg_id"))
            if pos is None or base is None:
                continue
            t = base + pos
            zones.append((t - NO_CUT_ZONE_PAD_S, t + NO_CUT_ZONE_PAD_S))
    return merge_close(zones, 0.0)


def run(job_dir: Path, report) -> None:
    vid = job_dir.name
    speakers = read_jsonl(job_dir / "manifests" / "speakers.jsonl")
    if not speakers:
        raise RuntimeError("speakers.jsonl missing — run diarize first")

    zones = _no_cut_zones(job_dir)
    if zones:
        report(f"{len(zones)} no-cut zones from prior detector runs", 0.02)

    report("loading audio", 0.05)
    original = load_audio(job_dir / "audio" / "original.wav", SR)

    all_turns = {r["speaker"]: [(t["start"], t["end"]) for t in r["turns"]]
                 for r in speakers}
    rows = []
    seg_dir = job_dir / "segments"
    for si, spk_row in enumerate(speakers):
        spk = spk_row["speaker"]
        others = [(s - CROSSTALK_GUARD_S, e + CROSSTALK_GUARD_S)
                  for o, ts in all_turns.items() if o != spk for s, e in ts]

        # If a clean lane exists for this speaker, cut from it and KEEP the
        # overlap seconds that separation rescued (purity-passing windows) —
        # that is the whole point of separating before transcribing. Where
        # rescue failed or was never attempted, fall back to the original
        # policy: subtract other speakers (crosstalk is TTS poison).
        lane_wav, engine, passing = _clean_lane_for(job_dir, spk)
        if lane_wav is not None:
            audio = load_audio(lane_wav, SR)
            keep = [(t["start"], t["end"]) for t in spk_row["turns"]]
            remove = subtract_intervals(others, passing)
            source = f"clean:{engine}"
        else:
            audio = original
            keep = [(t["start"], t["end"]) for t in spk_row["exclusive_turns"]]
            remove = others
            passing = []
            source = "original"

        clean = merge_close(subtract_intervals(keep, remove), MERGE_GAP_S)

        n_spk = 0
        for s, e in clean:
            if e - s < MIN_SEG_S:
                continue
            for cs, ce in split_long(audio, s, e, zones):
                trimmed = energy_trim(audio, cs, ce)
                if trimmed is None:
                    continue
                ts, te = trimmed
                if te - ts < MIN_SEG_S:
                    continue
                over = bool(te - ts > MAX_SEG_S)  # np.bool_ isn't JSON-serializable
                seg_id = f"{vid}_{spk}_{n_spk:04d}"
                dest = seg_dir / spk / f"{seg_id}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                i0, i1 = int(ts * SR), int(te * SR)
                sf.write(dest, audio[i0:i1], SR, subtype="PCM_16")
                # does this segment contain rescued (separated) audio?
                sep = any(a < te and b > ts for a, b in passing)
                rows.append({
                    "seg_id": seg_id,
                    "video_id": vid,
                    "speaker": spk,
                    "start": round(ts, 3),
                    "end": round(te, 3),
                    "dur": round(te - ts, 3),
                    "wav": f"segments/{spk}/{seg_id}.wav",
                    "over_len": over,
                    "source": source,        # original | clean:<engine>
                    "separated": bool(sep),  # contains rescued overlap audio
                })
                n_spk += 1
        n_sep = sum(1 for r in rows[-n_spk:] if r["separated"]) if n_spk else 0
        report(f"{spk}: {n_spk} segments from {source}"
               + (f" ({n_sep} contain rescued overlap)" if n_sep else ""),
               0.05 + 0.9 * (si + 1) / len(speakers))

    write_jsonl(job_dir / "manifests" / "segments.jsonl", rows)
