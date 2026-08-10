#!/usr/bin/env python3
"""Batch runner — the whole pipeline without the browser.

    # local audio files (or folders / globs)
    python run_batch.py calls/*.wav --detectors beats,laughs,tones,states

    # YouTube
    python run_batch.py "https://youtu.be/XXXX" --clean --detectors beats

    # collect every processed job into one corpus folder
    python run_batch.py --collect-only --out corpus/

Produces per job: jobs/<id>/manifests/dataset.jsonl (rows: clip + tagged text)
and, with --out, a portable corpus/ with the clips copied beside a combined
dataset.jsonl and a speaker x tag matrix.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from studio.config import JOBS_DIR, load_env                       # noqa: E402
from studio.detectors.registry import DETECTORS                    # noqa: E402
from studio.jobs import STAGE_ORDER, video_id_from_url             # noqa: E402
from studio.manifests import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from studio.stages import (s0_ingest, s1_diarize, s3_segment,      # noqa: E402
                           s4_asr, s5_align, s6_export)

STAGES = {"diarize": s1_diarize.run, "segment": s3_segment.run,
          "asr": s4_asr.run, "align": s5_align.run, "export": s6_export.run}


def log(msg: str) -> None:
    print(msg, flush=True)


def reporter(prefix: str):
    last = [""]

    def report(msg: str, progress: float | None = None) -> None:
        if msg and msg != last[0]:
            log(f"    {prefix}: {msg}")
            last[0] = msg
    return report


def job_for(src: str) -> tuple[str, str]:
    """(video_id, url) — url is '' for local files (upload mode)."""
    if src.startswith(("http://", "https://")):
        vid = video_id_from_url(src)
        if not vid:
            raise RuntimeError(f"cannot resolve a video id from {src}")
        return vid, src
    p = Path(src).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"no such file: {src}")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:6]
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in p.stem)[:40]
    vid = f"up_{stem}_{digest}"
    d = JOBS_DIR / vid
    d.mkdir(parents=True, exist_ok=True)
    if not list(d.glob("master.*")):
        shutil.copy2(p, d / f"master{p.suffix.lower()}")
    return vid, ""


def run_job(src: str, detectors: list[str], clean: bool, force: bool) -> str | None:
    vid, url = job_for(src)
    d = JOBS_DIR / vid
    d.mkdir(parents=True, exist_ok=True)
    log(f"\n=== {vid}  ({src})")

    marker = {"ingest": "manifests/stage0_video.json",
              "diarize": "manifests/speakers.jsonl",
              "segment": "manifests/segments.jsonl",
              "asr": "manifests/transcripts.jsonl",
              "align": "manifests/alignments.jsonl",
              "export": "manifests/dataset.jsonl"}

    for stage in STAGE_ORDER:
        done = (d / marker[stage]).exists()
        if done and not force:
            log(f"  · {stage:8s} (cached)")
            continue
        try:
            if stage == "ingest":
                s0_ingest.run(d, reporter(stage), url)
            else:
                # clean lanes must exist BEFORE segment cuts, so build here
                if stage == "segment" and clean:
                    build_clean_lanes(d)
                STAGES[stage](d, reporter(stage))
            log(f"  ✓ {stage}")
        except Exception as e:
            log(f"  ✗ {stage}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

        if stage == "align" and detectors:      # detectors feed the export
            for name in detectors:
                det = DETECTORS.get(name)
                if det is None:
                    log(f"  ✗ unknown detector {name}"); continue
                cand_file = d / "manifests" / f"candidates_{name}.jsonl"
                if cand_file.exists() and not force:
                    log(f"  · {name:8s} (cached)"); continue
                try:
                    segs = read_jsonl(d / "manifests" / "segments.jsonl")
                    rows = det.detect(d, segs, reporter(name))
                    write_jsonl(cand_file, rows)
                    log(f"  ✓ {name}: {len(rows)} candidates")
                except Exception as e:
                    log(f"  ✗ {name}: {type(e).__name__}: {e}")
    return vid


def build_clean_lanes(job_dir: Path) -> None:
    """SepFormer overlap rescue for every diarized speaker."""
    from studio.clean_lane import build_clean_lane
    speakers = read_jsonl(job_dir / "manifests" / "speakers.jsonl")
    for row in speakers:
        spk = row["speaker"]
        import re
        slug = re.sub(r"[^a-z0-9]+", "_", spk.lower())
        if (job_dir / "audio" / f"sam_{slug}_clean_sepformer.wav").exists():
            continue
        try:
            build_clean_lane(job_dir, reporter(f"clean/{spk}"), spk)
        except Exception as e:
            log(f"  ! clean lane {spk}: {type(e).__name__}: {e}")


def collect(out: Path, copy_audio: bool = True) -> None:
    """Merge every job's dataset.jsonl into one portable corpus."""
    out.mkdir(parents=True, exist_ok=True)
    rows, matrix = [], {}
    for jd in sorted(JOBS_DIR.iterdir()):
        ds = jd / "manifests" / "dataset.jsonl"
        if not ds.exists():
            continue
        for r in read_jsonl(ds):
            src = jd / r["wav"]
            if copy_audio and src.exists():
                dest = out / "clips" / r["speaker"] / f"{r['sample_id']}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(src, dest)
                r["wav"] = str(dest.relative_to(out))
            else:
                r["wav"] = str(src)
            rows.append(r)
            cell = matrix.setdefault(r["speaker"], {"_rows": 0, "_neutral": 0})
            cell["_rows"] += 1
            if not r["tags"]:
                cell["_neutral"] += 1
            for t in r["tags"]:
                cell[t] = cell.get(t, 0) + 1

    write_jsonl(out / "dataset.jsonl", rows)
    write_json(out / "matrix.json", matrix)
    dur = sum(r["dur"] for r in rows) / 3600
    log(f"\ncorpus: {len(rows)} rows · {dur:.2f} h · {len(matrix)} speakers -> {out}")
    log("\nspeaker x tag matrix (neutral rows matter as much as tagged ones):")
    tags = sorted({t for c in matrix.values() for t in c if not t.startswith("_")})
    log(f"  {'speaker':<18}{'rows':>6}{'neutral':>9}" + "".join(f"{t[:9]:>10}" for t in tags))
    for spk, c in sorted(matrix.items(), key=lambda kv: -kv[1]["_rows"]):
        log(f"  {spk[:18]:<18}{c['_rows']:>6}{c['_neutral']:>9}"
            + "".join(f"{c.get(t, 0):>10}" for t in tags))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="audio files, folders, globs or YouTube URLs")
    ap.add_argument("--detectors", default="",
                    help="comma list: " + ",".join(DETECTORS))
    ap.add_argument("--clean", action="store_true",
                    help="build SepFormer clean lanes before segmenting (overlap rescue)")
    ap.add_argument("--force", action="store_true", help="rerun cached stages")
    ap.add_argument("--out", type=Path, help="collect all jobs into this corpus dir")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip processing, just build the corpus from existing jobs")
    ap.add_argument("--no-copy", action="store_true",
                    help="corpus references clips in place instead of copying")
    args = ap.parse_args()

    load_env()
    dets = [d.strip() for d in args.detectors.split(",") if d.strip()]

    if not args.collect_only:
        srcs: list[str] = []
        for s in args.sources:
            if s.startswith(("http://", "https://")):
                srcs.append(s)
            elif Path(s).is_dir():
                srcs += [str(p) for p in sorted(Path(s).iterdir())
                         if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".webm",
                                                 ".flac", ".ogg", ".mp4"}]
            else:
                srcs += sorted(glob.glob(s)) or [s]
        if not srcs:
            ap.error("no sources given (or use --collect-only)")
        log(f"processing {len(srcs)} source(s); detectors: {dets or 'none'}"
            f"{'; clean lanes ON' if args.clean else ''}")
        ok = sum(1 for s in srcs if run_job(s, dets, args.clean, args.force))
        log(f"\n{ok}/{len(srcs)} jobs completed")

    if args.out:
        collect(args.out, copy_audio=not args.no_copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
