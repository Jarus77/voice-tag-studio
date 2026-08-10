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
        audio_np = d["audio"].astype(np.float32)
        lens = d["lens"]
        words = json.loads(bytes(d["words"]).decode())

        def emissions(x_np, ln):
            with torch.inference_mode():
                em, ol = self.model(
                    torch.from_numpy(x_np).to("cuda"),
                    torch.tensor(ln, device="cuda"))
            return em.cpu(), ol.cpu()

        try:
            em, out_lens = emissions(audio_np, lens)
            ems = [em[i, :int(out_lens[i])] for i in range(len(lens))]
        except torch.cuda.OutOfMemoryError:
            # chunks are up to 60 s — a full batch can exceed T4 memory;
            # fall back to one-at-a-time on this container
            torch.cuda.empty_cache()
            ems = []
            for i in range(len(lens)):
                e1, o1 = emissions(audio_np[i:i + 1, :int(lens[i])],
                                   lens[i:i + 1])
                ems.append(e1[0, :int(o1[0])])
                torch.cuda.empty_cache()

        out = []
        for i, ws in enumerate(words):
            try:
                spans = self.fa(ems[i], self.tokenizer(ws))
                ratio = float(lens[i]) / float(ems[i].size(0)) / 16000.0
                out.append([[s[0].start * ratio, s[-1].end * ratio]
                            for s in spans])
            except Exception as e:                       # per-chunk, not fatal
                out.append({"error": f"{type(e).__name__}: {str(e)[:120]}"})
        return json.dumps(out)
