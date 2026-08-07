"""SAM-Audio (Meta, promptable source separation) on Modal GPU.

Not a diarizer — it separates a described source ("man speaking",
"audience laughing") from the mix. In this studio it feeds LISTENING and
DETECTOR experiments only; separated audio never becomes dataset master
(generative separation can synthesize plausible audio — worse than demucs
artifacts, not better).

Usage (after accepting the gate at hf.co/facebook/sam-audio-base):
    modal run modal_sam_audio.py --job LhpZJwUboeI --start 900 --dur 60 \
        --prompt "man speaking"
    modal run modal_sam_audio.py --job LhpZJwUboeI --start 900 --dur 60 \
        --prompt "audience laughing" --model facebook/sam-audio-large

Outputs land as FULL-LENGTH timeline-aligned tracks (silence outside the
chunk) in jobs/<vid>/audio/sam_<prompt>_{target,residual}.wav — they appear
as extra lanes in the UI automatically.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import modal

APP_NAME = "tag-studio-sam-audio"
ROOT = pathlib.Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("git+https://github.com/facebookresearch/sam-audio")
    .env({"HF_HOME": "/cache/hf"})
)

app = modal.App(APP_NAME, image=image)
cache_vol = modal.Volume.from_name("voice-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")


@app.cls(
    gpu="L40S",
    volumes={"/cache": cache_vol},
    secrets=[hf_secret],
    timeout=1800,
    scaledown_window=120,
)
class Separator:
    @modal.enter()
    def setup(self) -> None:
        self._loaded: dict = {}

    def _get(self, model_name: str):
        if model_name not in self._loaded:
            from sam_audio import SAMAudio, SAMAudioProcessor

            model = SAMAudio.from_pretrained(model_name).eval().cuda()
            processor = SAMAudioProcessor.from_pretrained(model_name)
            self._loaded[model_name] = (model, processor)
            cache_vol.commit()
        return self._loaded[model_name]

    @modal.method()
    def separate(self, wav_bytes: bytes, description: str,
                 model_name: str = "facebook/sam-audio-base",
                 reranking: int = 1, predict_spans: bool = False) -> dict:
        self.model, self.processor = self._get(model_name)
        import io
        import tempfile

        import torch
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(wav_bytes)
            f.flush()
            batch = self.processor(
                audios=[f.name], descriptions=[description]).to("cuda")
            with torch.inference_mode():
                res = self.model.separate(
                    batch, predict_spans=predict_spans,
                    reranking_candidates=reranking)

        sr = self.processor.audio_sampling_rate

        def enc(t: "torch.Tensor") -> bytes:
            buf = io.BytesIO()
            x = t.cpu()
            if x.dim() == 1:
                x = x[None]
            torchaudio.save(buf, x, sr, format="wav")
            return buf.getvalue()

        return {"target": enc(res.target), "residual": enc(res.residual), "sr": sr}


@app.local_entrypoint()
def main(job: str, start: float = 900.0, dur: float = 60.0,
         prompt: str = "man speaking",
         model: str = "facebook/sam-audio-base",
         reranking: int = 1) -> None:
    import io
    from math import gcd

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    job_dir = ROOT / "jobs" / job
    master = next(job_dir.glob("master.*"), None)
    if master is None:
        raise SystemExit(f"no master audio in {job_dir}")

    chunk = job_dir / "_sam_chunk.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
         "-i", str(master), "-ac", "1", str(chunk)], check=True)

    print(f"separating {dur:.0f}s @ {start:.0f}s with {model!r}, prompt={prompt!r} ...")
    sep = Separator()
    out = sep.separate.remote(chunk.read_bytes(), prompt,
                              model_name=model, reranking=reranking)
    chunk.unlink()

    # place results on the full job timeline (silence elsewhere) so the UI's
    # shared-timeline lanes stay honest
    total = sf.info(job_dir / "audio" / "original.wav").frames
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:24]
    for kind in ("target", "residual"):
        data, sr = sf.read(io.BytesIO(out[kind]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        g = gcd(16000, sr)
        y = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
        track = np.zeros(total, dtype=np.float32)
        i0 = int(start * 16000)
        i1 = min(total, i0 + len(y))
        track[i0:i1] = y[: i1 - i0]
        dest = job_dir / "audio" / f"sam_{slug}_{kind}.wav"
        sf.write(dest, track, 16000, subtype="PCM_16")
        print("wrote", dest.name)
    print("open the UI — the sam_* lanes are listed next to original/denoised")
