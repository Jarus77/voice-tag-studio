"""Modal GPU backend for SepFormer overlap separation.

Deploy once:

    modal deploy modal_sepformer.py

The clean step auto-detects the deployed app (studio/clean_lane.py::
_separate_all) and falls back to batched local CPU when it's missing or
unreachable, so this file is optional infrastructure — nothing else imports it.

Payload contract (both directions float16 npz, ~5 MB per 8-window chunk):
    in : mixes (B, T)      window waveforms, zero-padded to the chunk max
    out: est   (B, T, 2)   both separated streams per window
Stream assignment / purity gating stay on the client — this function is pure
separation compute.
"""

import io

import modal

MODEL = "speechbrain/sepformer-whamr16k"
SAVEDIR = "/models/sepformer-whamr16k"

app = modal.App("voice-tag-sepformer")


def _download() -> None:
    from speechbrain.inference.separation import SepformerSeparation
    SepformerSeparation.from_hparams(source=MODEL, savedir=SAVEDIR,
                                     run_opts={"device": "cpu"})


image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "torchaudio", "speechbrain==1.1.0", "numpy")
         .run_function(_download))


@app.cls(image=image, gpu="T4", timeout=600, scaledown_window=120,
         max_containers=8)
class Separator:
    @modal.enter()
    def load(self) -> None:
        from speechbrain.inference.separation import SepformerSeparation
        self.sep = SepformerSeparation.from_hparams(
            source=MODEL, savedir=SAVEDIR, run_opts={"device": "cuda"})

    @modal.method()
    def separate(self, blob: bytes) -> bytes:
        import numpy as np
        import torch
        d = np.load(io.BytesIO(blob))
        mixes = torch.from_numpy(d["mixes"].astype(np.float32))
        with torch.inference_mode():
            est = self.sep.separate_batch(mixes)          # (B, T, 2)
        out = io.BytesIO()
        np.savez_compressed(out, est=est.cpu().numpy().astype(np.float16))
        return out.getvalue()
