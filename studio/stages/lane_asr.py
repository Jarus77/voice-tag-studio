"""Stage — lane ASR: transcribe each speaker's LANE in large chunks.

Architecture inversion (2026-08-10): transcription now happens BEFORE
cutting. Old order cut clips blind to word positions, and a boundary landing
mid-phrase made the ASR silently drop the truncated tail ("Liquid से" case)
— a text↔audio mismatch inside a training clip.

Chunks are ~60s slices of the speaker's lane (clean lane when built, masked
track otherwise), with edges placed in LONG silences between utterance
blocks — so no chunk edge can truncate speech. Fewer, larger Gemini calls
also give the model context, which improves verbatim quality.

Output: manifests/lane_transcripts.jsonl
    {speaker, chunk_id, start, end, text, engine, error}
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from ..config import load_env
from ..intervals import merge_close
from ..manifests import read_json, read_jsonl, write_jsonl

CHUNK_MAX_S = 60.0        # gemini handles this easily; srota capped lower
CHUNK_MAX_SROTA_S = 25.0  # qwen3-asr hard limit is 30s
CHUNK_BREAK_GAP_S = 4.0   # a gap this long always ends a chunk
BLOCK_MERGE_GAP_S = 1.0   # turns closer than this form one utterance block
CHUNK_PAD_S = 0.3         # breathing room at chunk edges (edges sit in silence)


def lane_wav_for(job_dir: Path, speaker: str) -> tuple[Path, str]:
    """The audio this speaker's transcription (and later cutting) uses:
    clean lane when built, silence-masked track otherwise."""
    slug = re.sub(r"[^a-z0-9]+", "_", speaker.lower())
    for name, src in ((f"sam_{slug}_clean_sepformer", "clean:sepformer"),
                      (f"sam_{slug}_clean_sam", "clean:sam"),
                      (f"sam_{slug}_clean", "clean:legacy")):
        p = job_dir / "audio" / f"{name}.wav"
        if p.exists():
            return p, src
    return job_dir / "audio" / "speakers" / f"{speaker}.wav", "masked"


def chunks_for(turns: list[tuple], max_s: float) -> list[tuple]:
    """Greedy chunking of utterance blocks; edges always in long silences."""
    blocks = merge_close(turns, BLOCK_MERGE_GAP_S)
    chunks: list[tuple] = []
    cur: list[float] | None = None
    for a, b in blocks:
        if cur is None:
            cur = [a, b]
            continue
        gap = a - cur[1]
        if (b - cur[0] > max_s) or (gap >= CHUNK_BREAK_GAP_S and cur[1] - cur[0] >= 20):
            chunks.append((cur[0], cur[1]))
            cur = [a, b]
        else:
            cur[1] = b
    if cur:
        chunks.append((cur[0], cur[1]))
    # a single unbroken block longer than max_s stays whole for gemini —
    # correctness (no mid-speech edge) beats the soft size preference
    return chunks


def _cut(lane: Path, a: float, b: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{max(0.0, a - CHUNK_PAD_S):.3f}",
         "-t", f"{(b - a) + 2 * CHUNK_PAD_S:.3f}",
         "-i", str(lane), "-c:a", "pcm_s16le", str(dest)], check=True)


def run(job_dir: Path, report) -> None:
    load_env()
    speakers = read_jsonl(job_dir / "manifests" / "speakers.jsonl")
    if not speakers:
        raise RuntimeError("speakers.jsonl missing — run diarize first")

    engine = read_json(job_dir / "job.json").get("asr_engine", "gemini")
    if engine == "sarvam":
        raise RuntimeError("sarvam can't do lane chunks (30s cap + drops "
                           "fillers) — use gemini or srota")

    # plan all chunks first
    max_s = CHUNK_MAX_SROTA_S if engine == "srota" else CHUNK_MAX_S
    plan = []      # (speaker, chunk_id, start, end, lane, source)
    for row in speakers:
        spk = row["speaker"]
        lane, source = lane_wav_for(job_dir, spk)
        turns = [(t["start"], t["end"]) for t in row["turns"]]
        for k, (a, b) in enumerate(chunks_for(turns, max_s)):
            plan.append((spk, f"{spk}_c{k:03d}", a, b, lane, source))
    report(f"{len(plan)} chunks across {len(speakers)} speakers ({engine})", 0.02)

    tmpdir = Path(tempfile.mkdtemp(prefix="lane_asr_", dir=job_dir))
    rows = []
    try:
        if engine == "srota":
            rows = _run_srota(job_dir, plan, tmpdir, report)
        else:
            rows = _run_gemini(plan, tmpdir, report)
    finally:
        for f in tmpdir.glob("*"):
            f.unlink(missing_ok=True)
        tmpdir.rmdir()

    write_jsonl(job_dir / "manifests" / "lane_transcripts.jsonl", rows)
    n_err = sum(1 for r in rows if r.get("error"))
    if rows and n_err == len(rows):
        raise RuntimeError("every chunk failed ASR — check key/quota")
    words = sum(len((r.get("text") or "").split()) for r in rows)
    report(f"done: {len(rows)} chunks, ~{words} words, {n_err} errors", 1.0)


def _run_gemini(plan, tmpdir: Path, report) -> list[dict]:
    import os
    from concurrent.futures import ThreadPoolExecutor

    from ..asr_bakeoff import _gemini_transcribe

    # network-bound, so concurrency is nearly free speed — bounded only by
    # the Gemini rate limit (Flash paid tier takes 16 easily)
    workers = int(os.environ.get("VTS_ASR_CONCURRENCY", "16"))

    def one(item):
        spk, cid, a, b, lane, source = item
        wav = tmpdir / f"{cid}.wav"
        _cut(lane, a, b, wav)
        r = {}
        for attempt in range(4):          # backoff absorbs 429s at high concurrency
            r = _gemini_transcribe(wav)
            if not r.get("error") and (r.get("text") or "").strip():
                break
            time.sleep(2 ** attempt)
        wav.unlink(missing_ok=True)
        return {"speaker": spk, "chunk_id": cid,
                "start": round(a, 3), "end": round(b, 3),
                "text": (r.get("text") or "").strip(),
                "engine": "gemini:lane", "source": source,
                "error": r.get("error")}

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, plan):
            rows.append(row)
            done += 1
            report(f"gemini chunk {done}/{len(plan)} ({workers} parallel)",
                   0.05 + 0.9 * done / len(plan))
    return rows


def _run_srota(job_dir: Path, plan, tmpdir: Path, report) -> list[dict]:
    import json as _json
    import subprocess as sp

    from ..config import ASR_MODEL, ASR_VENV_PY, ASR_WORKER
    if not ASR_VENV_PY.exists():
        raise RuntimeError(".venv-asr missing — see README setup")

    for spk, cid, a, b, lane, source in plan:
        _cut(lane, a, b, tmpdir / f"{cid}.wav")

    batch = tmpdir / "batch.json"
    out = tmpdir / "out.jsonl"
    batch.write_text(_json.dumps({
        "model": ASR_MODEL, "device": "mps",
        "segments": [{"seg_id": cid, "wav": str(tmpdir / f"{cid}.wav")}
                     for _, cid, *_ in plan]}))
    proc = sp.Popen([str(ASR_VENV_PY), str(ASR_WORKER),
                     "--batch", str(batch), "--out", str(out)],
                    stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    while proc.poll() is None:
        time.sleep(2)
        n = len(out.read_text().splitlines()) if out.exists() else 0
        report(f"srota chunk {n}/{len(plan)}", 0.05 + 0.9 * n / max(1, len(plan)))
    got = {r["seg_id"]: r for r in read_jsonl(out)}
    return [{"speaker": spk, "chunk_id": cid,
             "start": round(a, 3), "end": round(b, 3),
             "text": (got.get(cid, {}).get("text") or "").strip(),
             "engine": "srota:lane", "source": source,
             "error": got.get(cid, {}).get("error")}
            for spk, cid, a, b, lane, source in plan]
