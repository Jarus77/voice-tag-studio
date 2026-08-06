"""ASR worker — runs under .venv-asr (qwen-asr pins transformers==4.57.6,
which is incompatible with the main env's pyannote stack; hence this process
boundary).

Loads srota ONCE, transcribes a batch of segment wavs, streams one JSONL row
per segment to --out so the parent can report progress by counting lines.

Usage:
    .venv-asr/bin/python workers/asr_worker.py --batch batch.json --out out.jsonl
batch.json: {"model": "...", "device": "mps", "segments": [{"seg_id", "wav"}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHUNK = 8  # wavs per transcribe() call — bounded memory, steady progress


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def load_model(model_id: str, device: str):
    from qwen_asr import Qwen3ASRModel
    try:
        m = Qwen3ASRModel.from_pretrained(model_id, device_map=device, dtype="auto")
        log(f"loaded {model_id} on {device}")
        return m, device
    except Exception as e:
        if device != "cpu":
            log(f"{device} load failed ({type(e).__name__}: {e}) — retrying on cpu")
            m = Qwen3ASRModel.from_pretrained(model_id, device_map="cpu", dtype="auto")
            return m, "cpu"
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.batch.read_text())
    segments = spec["segments"]
    model, device = load_model(spec.get("model", "Surajgameramp/srota"),
                               spec.get("device", "mps"))

    with args.out.open("w") as f:
        for i in range(0, len(segments), CHUNK):
            chunk = segments[i:i + CHUNK]
            wavs = [c["wav"] for c in chunk]
            try:
                # language=None -> code-switched decoding (Hinglish)
                results = model.transcribe(wavs, language=None)
            except Exception as e:
                # first MPS *inference* failure: reload on cpu and retry once
                if device != "cpu":
                    log(f"inference failed on {device} ({type(e).__name__}) — cpu fallback")
                    model, device = load_model(spec.get("model", "Surajgameramp/srota"), "cpu")
                    try:
                        results = model.transcribe(wavs, language=None)
                    except Exception as e2:
                        results = None
                        err = f"{type(e2).__name__}: {str(e2)[:200]}"
                else:
                    results = None
                    err = f"{type(e).__name__}: {str(e)[:200]}"
            for j, c in enumerate(chunk):
                if results is None:
                    row = {"seg_id": c["seg_id"], "error": err, "device": device}
                else:
                    r = results[j]
                    row = {"seg_id": c["seg_id"],
                           "text": (r.text or "").strip(),
                           "language": r.language or "",
                           "device": device}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            log(f"{min(i + CHUNK, len(segments))}/{len(segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
