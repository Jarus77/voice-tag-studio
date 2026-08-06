"""Job model: one dir per video, resumable stages, single runner thread.

A stage is DONE iff its marker artifact exists. Rerun with force deletes the
marker first. Status lives in memory and is mirrored to jobs/<vid>/job.json so
restarts recover.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import traceback
from pathlib import Path

from .config import JOBS_DIR
from .manifests import read_json, write_json

STAGE_ORDER = ["ingest", "diarize", "denoise", "segment", "asr", "align"]

# stage -> marker path (relative to job dir) proving the stage completed
MARKERS = {
    "ingest": "manifests/stage0_video.json",
    "diarize": "manifests/speakers.jsonl",
    "denoise": "audio/denoised.wav",
    "segment": "manifests/segments.jsonl",
    "asr": "manifests/transcripts.jsonl",
    "align": "manifests/alignments.jsonl",
}

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})")


def video_id_from_url(url: str) -> str | None:
    m = _YT_ID_RE.search(url)
    if m:
        return m.group(1)
    # fallback: ask yt-dlp (network, ~1s)
    try:
        out = subprocess.run(["yt-dlp", "--no-playlist", "--print", "id", url],
                             capture_output=True, text=True, timeout=30)
        vid = out.stdout.strip().splitlines()[-1] if out.returncode == 0 else ""
        return vid or None
    except Exception:
        return None


class JobStore:
    """In-memory job status mirrored to job.json; one background runner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: dict[str, dict] = {}          # vid -> job.json contents
        self._queue: queue.Queue = queue.Queue()    # (vid, stage_name, fn)
        self._busy: str | None = None               # vid currently running
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._load_existing()

    # ---------- persistence ----------

    def _load_existing(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        for d in JOBS_DIR.iterdir():
            jf = d / "job.json"
            if d.is_dir() and jf.exists():
                job = read_json(jf)
                # anything marked running when the server died is stale
                for st in job.get("stages", {}).values():
                    if st.get("status") == "running":
                        st["status"] = "error"
                        st["msg"] = "interrupted (server restart)"
                self._status[d.name] = job
                self._refresh_done(d.name)

    def _persist(self, vid: str) -> None:
        write_json(JOBS_DIR / vid / "job.json", self._status[vid])

    def _refresh_done(self, vid: str) -> None:
        """Reconcile stage status with on-disk markers (resume logic)."""
        job = self._status[vid]
        for stage, marker in MARKERS.items():
            st = job["stages"].setdefault(stage, {"status": "pending"})
            if (JOBS_DIR / vid / marker).exists():
                if st.get("status") != "done":
                    st.update(status="done", msg="", progress=1.0)
            elif st.get("status") == "done":
                st.update(status="pending", msg="", progress=0.0)

    # ---------- public API ----------

    def create(self, vid: str, url: str, title: str = "") -> dict:
        with self._lock:
            if vid not in self._status:
                self._status[vid] = {
                    "video_id": vid, "url": url, "title": title,
                    "stages": {s: {"status": "pending"} for s in STAGE_ORDER},
                    "detectors": {},
                }
                (JOBS_DIR / vid).mkdir(parents=True, exist_ok=True)
            self._refresh_done(vid)
            self._persist(vid)
            return self._status[vid]

    def get(self, vid: str) -> dict | None:
        with self._lock:
            if vid not in self._status:
                return None
            self._refresh_done(vid)
            return json.loads(json.dumps(self._status[vid]))  # deep copy

    def list(self) -> list[dict]:
        with self._lock:
            return [{"video_id": v, "title": j.get("title", ""),
                     "url": j.get("url", "")} for v, j in self._status.items()]

    def busy(self) -> str | None:
        return self._busy

    def set_field(self, vid: str, key: str, value) -> None:
        with self._lock:
            self._status[vid][key] = value
            self._persist(vid)

    def set_stage(self, vid: str, stage: str, **fields) -> None:
        with self._lock:
            self._status[vid]["stages"].setdefault(stage, {}).update(fields)
            self._persist(vid)

    def set_detector(self, vid: str, name: str, **fields) -> None:
        with self._lock:
            self._status[vid].setdefault("detectors", {}).setdefault(name, {}).update(fields)
            self._persist(vid)

    def enqueue_stage(self, vid: str, stage: str, fn, force: bool = False) -> bool:
        """fn(job_dir, report) runs on the runner thread. Returns False if the
        stage is already done and force is not set."""
        job_dir = JOBS_DIR / vid
        marker = job_dir / MARKERS.get(stage, f"manifests/{stage}.done")
        if marker.exists():
            if not force:
                return False
            marker.unlink()
        self.set_stage(vid, stage, status="queued", msg="", progress=0.0)
        self._queue.put((vid, stage, fn))
        return True

    def enqueue_detect(self, vid: str, name: str, fn) -> None:
        self.set_detector(vid, name, status="queued", msg="")
        self._queue.put((vid, f"detect:{name}", fn))

    # ---------- runner ----------

    def _loop(self) -> None:
        while True:
            vid, stage, fn = self._queue.get()
            self._busy = vid
            is_det = stage.startswith("detect:")
            det_name = stage.split(":", 1)[1] if is_det else ""

            def report(msg: str, progress: float | None = None) -> None:
                fields = {"msg": str(msg)[:300]}
                if progress is not None:
                    fields["progress"] = round(float(progress), 3)
                if is_det:
                    self.set_detector(vid, det_name, **fields)
                else:
                    self.set_stage(vid, stage, **fields)

            try:
                if is_det:
                    self.set_detector(vid, det_name, status="running", msg="")
                else:
                    self.set_stage(vid, stage, status="running", msg="", progress=0.0)
                fn(JOBS_DIR / vid, report)
                if is_det:
                    self.set_detector(vid, det_name, status="done", msg="")
                else:
                    self.set_stage(vid, stage, status="done", msg="", progress=1.0)
            except Exception as e:
                traceback.print_exc()
                err = f"{type(e).__name__}: {str(e)[:250]}"
                if is_det:
                    self.set_detector(vid, det_name, status="error", msg=err)
                else:
                    self.set_stage(vid, stage, status="error", msg=err)
            finally:
                self._busy = None
                self._queue.task_done()
