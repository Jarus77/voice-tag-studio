"""Stage 4 — Transcribe: srota (qwen3-asr Hinglish fine-tune) via the
.venv-asr subprocess worker; Sarvam REST retries error rows.

Text convention: verbatim codemix — Devanagari Hindi + Latin loanwords, the
same discipline as the voice project's TTS_INPUT_SPEC.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import unicodedata
from pathlib import Path

from ..config import (ASR_MODEL, ASR_VENV_PY, ASR_WORKER, SARVAM_MODEL,
                      SARVAM_URL, load_env)
from ..manifests import read_jsonl, write_jsonl


def script_profile(text: str) -> dict:
    """Devanagari vs Latin composition, and how often the text switches."""
    deva = latin = other = 0
    seq = []
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            deva += 1
            seq.append("D")
        elif "a" <= ch.lower() <= "z":
            latin += 1
            seq.append("L")
        else:
            other += 1
    total = deva + latin + other
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return {
        "chars": total,
        "deva_frac": round(deva / total, 4) if total else 0.0,
        "latin_frac": round(latin / total, 4) if total else 0.0,
        "script_switches": switches,
    }


def sarvam_transcribe(wav: Path, key: str) -> dict:
    import requests
    try:
        with wav.open("rb") as fh:
            r = requests.post(
                SARVAM_URL,
                headers={"api-subscription-key": key},
                files={"file": (wav.name, fh, "audio/wav")},
                data={"model": SARVAM_MODEL, "mode": "codemix",
                      "language_code": "unknown"},
                timeout=180)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:160]}"}
        d = r.json()
        return {"text": d.get("transcript", ""),
                "language": d.get("language_code", "")}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}


def _run_sarvam_all(job_dir: Path, segments: list[dict], report) -> None:
    """Full-corpus Sarvam transcription (engine=sarvam). ~55 rpm rate cap."""
    import time
    load_env()
    key = os.environ.get("SARVAM_API_KEY", "")
    if not key:
        raise RuntimeError("SARVAM_API_KEY missing in .env")
    rows = []
    min_interval = 60.0 / 55.0
    last = 0.0
    for i, s in enumerate(segments):
        wait = min_interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        r = sarvam_transcribe(job_dir / s["wav"], key)
        text = (r.get("text") or "").strip()
        rows.append({
            "seg_id": s["seg_id"],
            "text": text,
            "asr": "sarvam:codemix",
            "language": r.get("language", ""),
            "script": script_profile(text) if text else None,
            "error": r.get("error"),
        })
        if (i + 1) % 10 == 0 or i + 1 == len(segments):
            report(f"sarvam: {i + 1}/{len(segments)}", (i + 1) / len(segments))
    write_jsonl(job_dir / "manifests" / "transcripts.jsonl", rows)
    n_err = sum(1 for r in rows if r["error"])
    if n_err == len(rows):
        raise RuntimeError("every segment failed Sarvam ASR — check key/credits")
    report(f"done, {n_err} errors", 1.0)


def run(job_dir: Path, report) -> None:
    segments = read_jsonl(job_dir / "manifests" / "segments.jsonl")
    if not segments:
        raise RuntimeError("segments.jsonl missing — run segment first")

    from ..manifests import read_json
    engine = read_json(job_dir / "job.json").get("asr_engine", "srota")
    if engine == "sarvam":
        _run_sarvam_all(job_dir, segments, report)
        return
    if not ASR_VENV_PY.exists():
        raise RuntimeError(".venv-asr missing — see README setup")

    batch_file = job_dir / "_asr_batch.json"
    out_file = job_dir / "_asr_out.jsonl"
    batch_file.write_text(json.dumps({
        "model": ASR_MODEL,
        "device": "mps",
        "segments": [{"seg_id": s["seg_id"], "wav": str(job_dir / s["wav"])}
                     for s in segments],
    }))
    if out_file.exists():
        out_file.unlink()

    report(f"srota worker: 0/{len(segments)}", 0.02)
    proc = subprocess.Popen(
        [str(ASR_VENV_PY), str(ASR_WORKER),
         "--batch", str(batch_file), "--out", str(out_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    stderr_tail: list[str] = []

    def _drain() -> None:  # keep the pipe from filling; keep last lines for errors
        for line in proc.stderr:
            stderr_tail.append(line.rstrip())
            del stderr_tail[:-20]

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    import time
    while proc.poll() is None:
        time.sleep(2.0)
        done = sum(1 for l in out_file.read_text().splitlines() if l.strip()) \
            if out_file.exists() else 0
        report(f"srota worker: {done}/{len(segments)}",
               0.02 + 0.85 * done / len(segments))
    t.join(timeout=5)
    if proc.returncode != 0:
        raise RuntimeError("asr worker failed: " + " | ".join(stderr_tail[-3:]))

    by_id = {r["seg_id"]: r for r in read_jsonl(out_file)}

    # Sarvam retries for worker-errored rows
    load_env()
    key = os.environ.get("SARVAM_API_KEY", "")
    errored = [s for s in segments if by_id.get(s["seg_id"], {}).get("error")]
    if errored and key:
        for i, s in enumerate(errored):
            report(f"sarvam fallback {i + 1}/{len(errored)}", 0.87 + 0.1 * i / len(errored))
            r = sarvam_transcribe(job_dir / s["wav"], key)
            if not r.get("error"):
                by_id[s["seg_id"]] = {"seg_id": s["seg_id"], **r, "device": "sarvam"}

    rows = []
    for s in segments:
        r = by_id.get(s["seg_id"], {"error": "no worker output"})
        text = (r.get("text") or "").strip()
        rows.append({
            "seg_id": s["seg_id"],
            "text": text,
            "asr": "sarvam:codemix" if r.get("device") == "sarvam" else f"srota:{r.get('device', '?')}",
            "language": r.get("language", ""),
            "script": script_profile(text) if text else None,
            "error": r.get("error"),
        })
    write_jsonl(job_dir / "manifests" / "transcripts.jsonl", rows)
    batch_file.unlink(missing_ok=True)
    out_file.unlink(missing_ok=True)

    n_err = sum(1 for r in rows if r["error"])
    if n_err == len(rows):
        raise RuntimeError("every segment failed ASR — check worker env")
    report(f"done, {n_err} errors", 1.0)
