"""Phase 1 — ASR bake-off: srota vs Sarvam vs Whisper-large-v3 vs Gemini.

Qualitative screening per DETECTOR_EVAL_PLAN.md: same segments through every
engine, side-by-side in the UI, fillers highlighted, misses counted. Judged
by ear+eye; the gold-set quantification comes later (Phase 0').
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .config import ASR_MODEL, ASR_VENV_PY, ASR_WORKER, load_env
from .manifests import read_jsonl, write_jsonl
from .stages.s4_asr import sarvam_transcribe

ENGINES = ["srota", "sarvam", "gemini"]   # whisper dropped by user's call

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_PROMPT = """Transcribe this Hindi/Hinglish (code-mixed) audio VERBATIM.

SCRIPT RULE — this is the most important instruction:
• Hindi words → Devanagari.
• English words → LATIN script, always. NEVER transliterate an English word
  into Devanagari, no matter how commonly it is spoken in Hindi.
• This includes brand names, products, finance/tech terms and abbreviations.

Correct:  आप ICICI में trades phone में लेते हो या laptop PC में?
WRONG:    आप आई सी आई सी में ट्रेड्स फ़ोन में लेते हो या लैपटॉप पीसी में?

Correct:  And returns की बात करें तो quarter में 87,000 का profit बनाया था
WRONG:    एंड रिटर्न्स की बात करें तो क्वार्टर में 87,000 का प्रॉफिट बनाया था

Correct:  हमारा focus purely quality trade and consistency पे है
WRONG:    हमारा फोकस प्योरली क्वालिटी ट्रेड एंड कंसिस्टेंसी पे है

VERBATIM RULE:
• Keep every filler exactly as spoken (उह, अह, हम्म, अं, हाँ, मतलब, यानी,
  uh, um, hmm), plus repetitions, false starts and incomplete words.
• Do not clean up, summarise, translate or reorder anything.

Output ONLY the transcript text — no preamble, no notes."""

def _run_srota(job_dir: Path, segs: list[dict], report) -> dict[str, dict]:
    batch = job_dir / "_bakeoff_batch.json"
    out = job_dir / "_bakeoff_out.jsonl"
    batch.write_text(json.dumps({
        "model": ASR_MODEL, "device": "mps",
        "segments": [{"seg_id": s["seg_id"], "wav": str(job_dir / s["wav"])}
                     for s in segs]}))
    out.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [str(ASR_VENV_PY), str(ASR_WORKER), "--batch", str(batch), "--out", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        time.sleep(2)
        n = len(out.read_text().splitlines()) if out.exists() else 0
        report(f"srota {n}/{len(segs)}", None)
    rows = {r["seg_id"]: r for r in read_jsonl(out)}
    batch.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return rows


def _gemini_transcribe(wav: Path) -> dict:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return {"error": "GEMINI_API_KEY missing in .env"}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[GEMINI_PROMPT,
                      types.Part.from_bytes(data=wav.read_bytes(),
                                            mime_type="audio/wav")])
        return {"text": (r.text or "").strip()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def run_bakeoff(job_dir: Path, report, limit: int = 12) -> None:
    load_env()
    segments = read_jsonl(job_dir / "manifests" / "segments.jsonl")
    if not segments:
        raise RuntimeError("segments.jsonl missing — run the segment stage first")
    if len(segments) > limit:   # evenly spaced sample across the audio
        step = len(segments) / limit
        segments = [segments[int(i * step)] for i in range(limit)]

    results = {s["seg_id"]: {"seg_id": s["seg_id"], "start": s["start"],
                             "dur": s["dur"], "wav": s["wav"],
                             "speaker": s["speaker"], "engines": {}}
               for s in segments}

    # srota (isolated venv worker, batch)
    report(f"srota on {len(segments)} segments", 0.05)
    t0 = time.time()
    srota = _run_srota(job_dir, segments, report)
    for s in segments:
        r = srota.get(s["seg_id"], {})
        results[s["seg_id"]]["engines"]["srota"] = {
            "text": (r.get("text") or "").strip(), "error": r.get("error")}
    report(f"srota done ({time.time() - t0:.0f}s)", 0.3)

    # sarvam (REST)
    key = os.environ.get("SARVAM_API_KEY", "")
    for i, s in enumerate(segments):
        r = sarvam_transcribe(job_dir / s["wav"], key) if key \
            else {"error": "SARVAM_API_KEY missing"}
        results[s["seg_id"]]["engines"]["sarvam"] = {
            "text": (r.get("text") or "").strip(), "error": r.get("error")}
        report(f"sarvam {i + 1}/{len(segments)}", 0.3 + 0.15 * (i + 1) / len(segments))

    # gemini (verbatim-prompted)
    for i, s in enumerate(segments):
        r = _gemini_transcribe(job_dir / s["wav"])
        results[s["seg_id"]]["engines"]["gemini"] = {
            "text": (r.get("text") or "").strip(), "error": r.get("error")}
        report(f"gemini {i + 1}/{len(segments)}",
               0.5 + 0.45 * (i + 1) / len(segments))

    rows = [results[s["seg_id"]] for s in segments]
    write_jsonl(job_dir / "manifests" / "asr_bakeoff.jsonl", rows)
    miss = {e: sum(1 for r in rows
                   if not r["engines"][e]["text"]) for e in ENGINES}
    report("done — misses: " + ", ".join(f"{e}:{n}" for e, n in miss.items()), 1.0)
