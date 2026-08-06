"""Detector contract.

A detector consumes a job's segments and emits CANDIDATE rows — nothing is
final here; the ear (audition) and later an audio-LLM judge tier decide.
Candidate rows follow the podcast_pipeline stage5_candidates shape:

    {seg_id, tag, score, detector, position_s|null, evidence: {...}}

position_s is SEGMENT-relative. Detectors declare which audio source they
read; default "original" — denoising strips exactly the non-speech vocal
events we tag.
"""

from __future__ import annotations

from pathlib import Path


class Detector:
    name: str = ""
    tags: list[str] = []
    source: str = "original"   # "original" (master) | "denoised"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        raise NotImplementedError
