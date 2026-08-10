"""voice-tag-studio server.

    python server.py            # http://127.0.0.1:8765

FastAPI + one background runner thread (single concurrent job). Stages are
resumable: done iff their manifest exists; rerun with ?force=1.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from studio.config import (ASR_VENV_PY, DETECTOR_SOURCE_DEFAULT,
                           DIARIZATION_MODEL, JOBS_DIR, PORT, STATIC_DIR,
                           load_env)
from studio.detectors.registry import DETECTORS
from studio.jobs import MARKERS, STAGE_ORDER, JobStore, video_id_from_url
from studio.manifests import read_json, read_jsonl, write_jsonl
from studio.stages import (s0_ingest, s1_diarize, s2_clean, s3_segment,
                          s4_asr, s5_align, s6_export)

load_env()
app = FastAPI(title="voice-tag-studio")
store = JobStore()

STAGE_FNS = {
    "ingest": None,  # bound per-job with its URL below
    "diarize": s1_diarize.run,
    "clean": s2_clean.run,
    "segment": s3_segment.run,
    "asr": s4_asr.run,
    "align": s5_align.run,
    "export": s6_export.run,
}


# ---------------- preflight ----------------

@app.get("/api/preflight")
def preflight() -> dict:
    import os

    checks: dict[str, dict] = {}

    def add(name, ok, msg=""):
        checks[name] = {"ok": bool(ok), "msg": msg}

    add("ffmpeg", shutil.which("ffmpeg"), "brew install ffmpeg")
    add("yt-dlp", shutil.which("yt-dlp"), "pip install yt-dlp")
    try:
        from huggingface_hub import get_token
        add("hf_token", bool(get_token()), "hf auth login")
    except Exception as e:
        add("hf_token", False, str(e)[:120])
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(DIARIZATION_MODEL, "config.yaml")
        add("pyannote_gate", True)
    except Exception as e:
        add("pyannote_gate", False,
            f"accept terms at https://huggingface.co/{DIARIZATION_MODEL} "
            f"(logged in as your HF account) — {type(e).__name__}")
    add("venv_asr", ASR_VENV_PY.exists(),
        "python3 -m venv .venv-asr && .venv-asr/bin/pip install -r requirements-asr.txt")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("Surajgameramp/srota", local_files_only=True,
                          allow_patterns=["config.json"])
        add("srota_cached", True)
    except Exception:
        add("srota_cached", False, "hf download Surajgameramp/srota")
    panns_ckpt = Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"
    add("panns_ckpt", panns_ckpt.exists(),
        "auto-downloads (~1.1GB) on first laughs run")
    try:
        import demucs  # noqa: F401
        add("demucs", True)
    except Exception:
        add("demucs", False, "pip install demucs (afftdn fallback still works)")
    add("sarvam_key", bool(os.environ.get("SARVAM_API_KEY")),
        "optional ASR fallback — put SARVAM_API_KEY in .env")

    return {"ok": all(c["ok"] for n, c in checks.items()
                      if n not in ("sarvam_key", "demucs", "panns_ckpt")),
            "checks": checks}


# ---------------- jobs ----------------

class NewJob(BaseModel):
    url: str


@app.post("/api/jobs")
def create_job(body: NewJob) -> dict:
    vid = video_id_from_url(body.url.strip())
    if not vid:
        raise HTTPException(400, "could not resolve a YouTube video id from that URL")
    job = store.create(vid, body.url.strip())
    # manual pipeline control: only ingest runs automatically (so there is
    # audio to inspect) — every other stage runs when its step is clicked
    store.enqueue_stage(
        vid, "ingest",
        lambda d, r, u=body.url.strip(): s0_ingest.run(d, r, u), force=False)
    return {"video_id": vid, "job": job}


@app.post("/api/upload")
async def upload_audio(file: UploadFile) -> dict:
    """Uploaded-audio jobs: same pipeline, no yt-dlp. Ideal for short test clips."""
    import hashlib
    import re as _re
    raw = await file.read()
    if len(raw) < 1000:
        raise HTTPException(400, "file too small / empty")
    stem = _re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file.filename or "upload").stem)[:40]
    vid = f"up_{stem}_{hashlib.sha256(raw).hexdigest()[:6]}"
    ext = Path(file.filename or "upload.wav").suffix.lower() or ".wav"
    job_dir = JOBS_DIR / vid
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"master{ext}").write_bytes(raw)
    store.create(vid, "")          # url="" -> upload mode in s0_ingest
    store.set_field(vid, "title", f"{stem} (upload)")
    # manual pipeline control: only ingest auto-runs
    store.enqueue_stage(vid, "ingest",
                        lambda d, r: s0_ingest.run(d, r, ""), force=False)
    return {"video_id": vid}


DIARIZERS = {
    "pyannote/speaker-diarization-community-1":
        "pyannote community-1 (local MPS, default)",
}


@app.get("/api/diarizers")
def list_diarizers() -> list[dict]:
    return [{"id": k, "label": v} for k, v in DIARIZERS.items()]


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return store.list()


@app.delete("/api/jobs/{vid}")
def delete_job(vid: str) -> dict:
    """Delete one job and everything it produced (audio, lanes, segments,
    manifests). Irreversible — jobs/ is gitignored."""
    import shutil as _sh
    if "/" in vid or ".." in vid:
        raise HTTPException(400, "bad job id")
    job_dir = JOBS_DIR / vid
    if not job_dir.exists():
        raise HTTPException(404, "no such job")
    if store.busy() == vid:
        raise HTTPException(409, "this job is running — wait for it to finish")
    size = sum(f.stat().st_size for f in job_dir.rglob("*") if f.is_file())
    _sh.rmtree(job_dir, ignore_errors=True)
    store.forget(vid)
    return {"deleted": vid, "freed_mb": round(size / 1e6, 1)}


@app.get("/api/jobs/{vid}")
def get_job(vid: str) -> dict:
    job = store.get(vid)
    if job is None:
        raise HTTPException(404, "no such job")
    job_dir = JOBS_DIR / vid
    meta = read_json(job_dir / "manifests" / "stage0_video.json")
    if meta.get("title") and not job.get("title"):
        store.set_field(vid, "title", meta["title"])
        job["title"] = meta["title"]
    # audio sources for the Listen section: original/denoised first, then any
    # extra full-length tracks dropped in audio/ (clean lanes),
    # then per-speaker masked tracks
    sources = []
    seen = set()
    for name in ["original", "denoised"]:
        if (job_dir / "audio" / f"{name}.wav").exists():
            sources.append({"name": name, "path": f"/files/{vid}/audio/{name}.wav"})
            seen.add(name)
    for wav in sorted((job_dir / "audio").glob("*.wav")):
        if wav.name.startswith("."):        # half-built lane, still writing
            continue
        if wav.stem not in seen:
            sources.append({"name": wav.stem, "path": f"/files/{vid}/audio/{wav.name}"})
    for spk in sorted((job_dir / "audio" / "speakers").glob("*.wav")):
        sources.append({"name": spk.stem,
                        "path": f"/files/{vid}/audio/speakers/{spk.name}"})
    job["sources"] = sources
    job["meta"] = meta
    job["detector_names"] = list(DETECTORS)
    job["busy"] = store.busy()
    job["activity"] = store.activity()
    job["overlaps"] = _overlaps(job_dir)
    # manifest mtimes let the UI notice a re-run and refetch (otherwise it
    # keeps showing the data it loaded the first time the stage finished)
    mt = {}
    md = job_dir / "manifests"
    if md.exists():
        for f in md.glob("*.json*"):
            mt[f.name] = int(f.stat().st_mtime)
    job["manifest_mtimes"] = mt
    return job


def _overlaps(job_dir: Path, min_s: float = 0.3, cap: int = 200) -> list[list[float]]:
    """Regions where >=2 diarized speakers talk at once (sweep line over turns).
    These are the crosstalk zones the segment stage drops — and the prime
    targets for SAM span-prompted rescue."""
    rows = read_jsonl(job_dir / "manifests" / "speakers.jsonl")
    if not rows:
        return []
    events = []
    for r in rows:
        for t in r["turns"]:
            events.append((t["start"], 1))
            events.append((t["end"], -1))
    events.sort()
    out, n, cur = [], 0, None
    for ts, delta in events:
        n += delta
        if n >= 2 and cur is None:
            cur = ts
        elif n < 2 and cur is not None:
            if ts - cur >= min_s:
                out.append([round(cur, 2), round(ts, 2)])
            cur = None
    return out[:cap]


@app.post("/api/jobs/{vid}/stages/{stage}")
def rerun_stage(vid: str, stage: str, force: bool = False,
                engine: str | None = None, guard: float | None = None) -> dict:
    if stage not in STAGE_ORDER:
        raise HTTPException(404, f"unknown stage {stage}")
    job = store.get(vid)
    if job is None:
        raise HTTPException(404, "no such job")
    if stage == "asr" and engine:
        if engine not in ("srota", "sarvam", "gemini"):
            raise HTTPException(400, "engine must be srota, sarvam or gemini")
        store.set_field(vid, "asr_engine", engine)
    if stage == "diarize" and engine:
        if engine not in DIARIZERS:
            raise HTTPException(400, f"unknown diarizer; see /api/diarizers")
        store.set_field(vid, "diarizer", engine)
    if stage == "segment" and guard is not None:
        if not 0.0 <= guard <= 1.0:
            raise HTTPException(400, "guard must be 0-1 seconds")
        store.set_field(vid, "crosstalk_guard", guard)
    if stage == "ingest":
        url = job.get("url", "")
        fn = lambda d, r: s0_ingest.run(d, r, url)  # noqa: E731
    else:
        fn = STAGE_FNS[stage]
    queued = store.enqueue_stage(vid, stage, fn, force=force)
    return {"queued": queued}


@app.get("/api/jobs/{vid}/manifests/{name}")
def get_manifest(vid: str, name: str) -> JSONResponse:
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = JOBS_DIR / vid / "manifests" / name
    if not path.exists():
        raise HTTPException(404, f"no manifest {name}")
    if name.endswith(".json"):
        return JSONResponse(read_json(path))
    return JSONResponse(read_jsonl(path))


@app.get("/api/jobs/{vid}/segpeaks/{seg_id}")
def get_segment_peaks(vid: str, seg_id: str) -> JSONResponse:
    """Mini-waveform for one segment (lazy, cached)."""
    from studio.audio import peaks_cached
    if "/" in seg_id or ".." in seg_id:
        raise HTTPException(400, "bad seg_id")
    row = next((r for r in read_jsonl(JOBS_DIR / vid / "manifests" / "segments.jsonl")
                if r["seg_id"] == seg_id), None)
    if row is None:
        raise HTTPException(404, "no such segment")
    wav = JOBS_DIR / vid / row["wav"]
    if not wav.exists():
        raise HTTPException(404, "segment wav missing")
    return JSONResponse(peaks_cached(wav, JOBS_DIR / vid / "peaks" / "segments", 160))


class SpeakerName(BaseModel):
    speaker: str
    name: str


@app.post("/api/jobs/{vid}/speaker_name")
def set_speaker_name(vid: str, body: SpeakerName) -> dict:
    """Name a diarized speaker (e.g. SPEAKER_00 -> 'khushi' / 'agent').
    pyannote labels are arbitrary — the dataset needs a real voice identity."""
    job = store.get(vid)
    if job is None:
        raise HTTPException(404, "no such job")
    names = dict(job.get("speaker_names") or {})
    clean = body.name.strip()[:40]
    if clean:
        names[body.speaker] = clean
    else:
        names.pop(body.speaker, None)
    store.set_field(vid, "speaker_names", names)
    return {"speaker_names": names}


@app.get("/api/jobs/{vid}/peaks/{name}")
def get_peaks(vid: str, name: str) -> JSONResponse:
    from studio.audio import peaks_cached
    job_dir = JOBS_DIR / vid
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    wav = job_dir / "audio" / f"{name}.wav"
    if not wav.exists():
        wav = job_dir / "audio" / "speakers" / f"{name}.wav"
    if not wav.exists():
        raise HTTPException(404, f"no audio source {name}")
    return JSONResponse(peaks_cached(wav, job_dir / "peaks"))


# ---------------- detectors ----------------

class DetectReq(BaseModel):
    detector: str
    source: str | None = None


@app.post("/api/jobs/{vid}/detect")
def run_detect(vid: str, body: DetectReq) -> dict:
    det = DETECTORS.get(body.detector)
    if det is None:
        raise HTTPException(404, f"unknown detector {body.detector}")
    job_dir = JOBS_DIR / vid
    segments = read_jsonl(job_dir / "manifests" / "segments.jsonl")
    if not segments:
        raise HTTPException(409, "run the segment stage first")

    def fn(d: Path, report) -> None:
        rows = det.detect(d, segments, report)
        write_jsonl(d / "manifests" / f"candidates_{det.name}.jsonl", rows)
        report(f"{len(rows)} candidates", 1.0)

    store.enqueue_detect(vid, det.name, fn)
    return {"queued": True, "detector": det.name,
            "source": body.source or det.source or DETECTOR_SOURCE_DEFAULT}


# ---------------- ASR bake-off (Phase 1) ----------------

class BakeoffReq(BaseModel):
    limit: int = 12


@app.post("/api/jobs/{vid}/bakeoff")
def run_asr_bakeoff(vid: str, body: BakeoffReq) -> dict:
    from studio.asr_bakeoff import run_bakeoff
    job_dir = JOBS_DIR / vid
    if not (job_dir / "manifests" / "segments.jsonl").exists():
        raise HTTPException(409, "run the segment stage first")
    lim = max(1, min(int(body.limit), 40))

    def fn(dir_: Path, report) -> None:
        run_bakeoff(dir_, report, limit=lim)

    store.enqueue_detect(vid, "bakeoff", fn)
    return {"queued": True}


# ---------------- clean-lane separation (SepFormer) ----------------

class SeparateReq(BaseModel):
    speaker: str
    full: bool = True
    engine: str = "sepformer"


@app.post("/api/jobs/{vid}/separate")
def run_separate(vid: str, body: SeparateReq) -> dict:
    """Build a clean full-length lane for one speaker: original where they
    speak alone, SepFormer separation at overlap windows, silence elsewhere."""
    from studio.clean_lane import build_clean_lane
    job_dir = JOBS_DIR / vid
    if not (job_dir / "audio" / "original.wav").exists():
        raise HTTPException(409, "run ingest first")
    if not (job_dir / "manifests" / "speakers.jsonl").exists():
        raise HTTPException(409, "run diarize first")
    spk = body.speaker

    def fn(dir_: Path, report) -> None:
        build_clean_lane(dir_, report, spk)

    store.enqueue_detect(vid, "separate", fn)
    return {"queued": True}


# ---------------- static ----------------

JOBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
