"""voice-tag-studio server.

    python server.py            # http://127.0.0.1:8765

FastAPI + one background runner thread (single concurrent job). Stages are
resumable: done iff their manifest exists; rerun with ?force=1.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from studio.config import (ASR_VENV_PY, DETECTOR_SOURCE_DEFAULT,
                           DIARIZATION_MODEL, JOBS_DIR, PORT, STATIC_DIR,
                           load_env)
from studio.detectors.registry import DETECTORS
from studio.jobs import MARKERS, STAGE_ORDER, JobStore, video_id_from_url
from studio.manifests import read_json, read_jsonl, write_jsonl
from studio.stages import s0_ingest, s1_diarize, s2_denoise, s3_segment, s4_asr, s5_align

load_env()
app = FastAPI(title="voice-tag-studio")
store = JobStore()

STAGE_FNS = {
    "ingest": None,  # bound per-job with its URL below
    "diarize": s1_diarize.run,
    "denoise": s2_denoise.run,
    "segment": s3_segment.run,
    "asr": s4_asr.run,
    "align": s5_align.run,
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
    # enqueue every not-yet-done stage in order
    for stage in STAGE_ORDER:
        fn = (lambda d, r, u=body.url.strip(): s0_ingest.run(d, r, u)) \
            if stage == "ingest" else STAGE_FNS[stage]
        store.enqueue_stage(vid, stage, fn, force=False)
    return {"video_id": vid, "job": job}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return store.list()


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
    # extra full-length tracks dropped in audio/ (e.g. sam_* separations),
    # then per-speaker masked tracks
    sources = []
    seen = set()
    for name in ["original", "denoised"]:
        if (job_dir / "audio" / f"{name}.wav").exists():
            sources.append({"name": name, "path": f"/files/{vid}/audio/{name}.wav"})
            seen.add(name)
    for wav in sorted((job_dir / "audio").glob("*.wav")):
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
    return job


@app.post("/api/jobs/{vid}/stages/{stage}")
def rerun_stage(vid: str, stage: str, force: bool = False,
                engine: str | None = None) -> dict:
    if stage not in STAGE_ORDER:
        raise HTTPException(404, f"unknown stage {stage}")
    job = store.get(vid)
    if job is None:
        raise HTTPException(404, "no such job")
    if stage == "asr" and engine:
        if engine not in ("srota", "sarvam"):
            raise HTTPException(400, "engine must be srota or sarvam")
        store.set_field(vid, "asr_engine", engine)
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


# ---------------- static ----------------

JOBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=JOBS_DIR), name="files")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
