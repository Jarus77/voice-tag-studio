"""Modal GPU backend for MMS forced alignment (the align stage's heavy half).

Deploy once:

    modal deploy modal_align.py

The align stage auto-detects the deployed app (studio/stages/lane_align.py::
_modal_spans) and falls back to local MPS/CPU when it's missing — this file is
optional infrastructure, nothing else imports it.

Split of labor: the client romanizes tokens (uroman) and runs VAD/island logic;
this function only computes wav2vec2 emissions on GPU and decodes word spans.

Payload contract:
    in : npz — audio (B, T) float16 zero-padded, lens (B,) int64,
         words: uint8 bytes of JSON list[list[str]] (romanized, per chunk)
    out: JSON list per chunk — [[start_s, end_s], ...] one per word,
         or {"error": ...} for a chunk whose decode failed
"""

import io
import json

import modal

app = modal.App("voice-tag-align")


def _download() -> None:
    import torchaudio
    b = torchaudio.pipelines.MMS_FA
    b.get_model(with_star=False)
    b.get_tokenizer()


image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "torchaudio", "numpy")
         .run_function(_download))


@app.cls(image=image, gpu="T4", timeout=600, scaledown_window=120,
         max_containers=8)
class Aligner:
    @modal.enter()
    def load(self) -> None:
        import torch
        import torchaudio
        self.torch = torch
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=False).to("cuda").eval()
        self.tokenizer = b.get_tokenizer()
        self.fa = b.get_aligner()

    @modal.method()
    def align(self, blob: bytes) -> str:
        import numpy as np
        torch = self.torch
        d = np.load(io.BytesIO(blob))
        audio = torch.from_numpy(d["audio"].astype(np.float32)).to("cuda")
        lens = d["lens"]
        words = json.loads(bytes(d["words"]).decode())
        with torch.inference_mode():
            em, out_lens = self.model(
                audio, torch.tensor(lens, device="cuda"))
        em, out_lens = em.cpu(), out_lens.cpu()
        out = []
        for i, ws in enumerate(words):
            try:
                spans = self.fa(em[i, :int(out_lens[i])], self.tokenizer(ws))
                ratio = float(lens[i]) / float(out_lens[i]) / 16000.0
                out.append([[s[0].start * ratio, s[-1].end * ratio]
                            for s in spans])
            except Exception as e:                       # per-chunk, not fatal
                out.append({"error": f"{type(e).__name__}: {str(e)[:120]}"})
        return json.dumps(out)
