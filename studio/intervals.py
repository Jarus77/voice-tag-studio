"""Interval math + energy trim + split (ported from voice pipeline stage4_segment).

One behavioral addition over the original: split_long() accepts no_cut_zones --
absolute-time regions that must never receive a cut. A [sigh]/[laughs] mid-turn
lives at exactly the quietest interior point, which is where the original
algorithm cuts; protecting those zones keeps tagged events EMBEDDED in flowing
speech (voice -> event -> same voice).
"""

from __future__ import annotations

import numpy as np

from .config import FRAME_S, MAX_SEG_S, MIN_SEG_S, PAD_S, SR


def subtract_intervals(keep: list[tuple], remove: list[tuple]) -> list[tuple]:
    """keep minus remove, both lists of (start, end) in seconds."""
    if not remove:
        return keep
    remove = sorted(remove)
    out = []
    for s, e in keep:
        cur = [(s, e)]
        for rs, re_ in remove:
            if re_ <= s or rs >= e:
                continue
            nxt = []
            for cs, ce in cur:
                if re_ <= cs or rs >= ce:
                    nxt.append((cs, ce))
                    continue
                if rs > cs:
                    nxt.append((cs, min(rs, ce)))
                if re_ < ce:
                    nxt.append((max(re_, cs), ce))
            cur = nxt
        out.extend(cur)
    return [(s, e) for s, e in out if e - s > 0]


def merge_close(intervals: list[tuple], gap: float) -> list[tuple]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((x[:n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))


def energy_trim(audio: np.ndarray, s: float, e: float) -> tuple[float, float] | None:
    """Snap [s,e] inward to actual speech onset/offset."""
    i0, i1 = int(s * SR), min(int(e * SR), len(audio))
    if i1 - i0 < int(MIN_SEG_S * SR):
        return None
    seg = audio[i0:i1]
    frame = int(FRAME_S * SR)
    rms = frame_rms(seg, frame)
    if len(rms) < 3:
        return None
    floor = np.percentile(rms, 10)
    thresh = max(floor * 3.0, rms.max() * 0.05)
    above = np.where(rms > thresh)[0]
    if len(above) == 0:
        return None
    ts = s + max(0.0, above[0] * FRAME_S - PAD_S)
    te = s + min((e - s), (above[-1] + 1) * FRAME_S + PAD_S)
    return (ts, te) if te - ts >= MIN_SEG_S else None


def split_long(audio: np.ndarray, s: float, e: float,
               no_cut_zones: list[tuple] = ()) -> list[tuple]:
    """Recursively split at the quietest interior frame until under MAX_SEG_S.

    Frames inside any no_cut_zone (absolute seconds) are never chosen as cut
    points. If every candidate frame is protected, the interval is returned
    un-split (callers flag over-length segments instead of bisecting an event).
    """
    if e - s <= MAX_SEG_S:
        return [(s, e)]
    i0, i1 = int(s * SR), min(int(e * SR), len(audio))
    seg = audio[i0:i1]
    frame = int(FRAME_S * SR)
    rms = frame_rms(seg, frame)
    if len(rms) < 10:
        return [(s, e)]
    # Only consider split points in the middle 60% so we don't shave a sliver.
    lo, hi = int(len(rms) * 0.2), int(len(rms) * 0.8)
    if hi <= lo:
        return [(s, e)]
    idx = np.arange(lo, hi)
    if no_cut_zones:
        times = s + idx * FRAME_S
        allowed = np.ones(len(idx), dtype=bool)
        for zs, ze in no_cut_zones:
            allowed &= ~((times >= zs) & (times <= ze))
        if not allowed.any():
            return [(s, e)]
        idx = idx[allowed]
    cut = int(idx[np.argmin(rms[idx])])
    mid = s + cut * FRAME_S
    return (split_long(audio, s, mid, no_cut_zones)
            + split_long(audio, mid, e, no_cut_zones))
