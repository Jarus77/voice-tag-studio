"""Emotional states — [excited] [nervous] [frustrated] [sorrowful] [calm].

audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim: dimensional
arousal/dominance/valence regression in [0,1]. English-trained — treat scores
as WEAK evidence on Hindi (evidence.weak=true); percentile space is
PER-SPEAKER (relative mode), never absolute. Region map:

  excited     A high, V high        nervous    A high, V low, unstable F0
  frustrated  A mid-high, V low     sorrowful  A low, V low
  calm        A low, V mid-high

The audio-LLM judge tier owns the final say on all five (not built yet);
until then these candidates exist to be auditioned, not trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detector
from .prosody import features, speaker_percentile

MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MIN_SEGS_PER_SPK = 20

_model = None  # lazy singleton (processor, model, device)


def _load():
    global _model
    if _model is None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        # The classic audeering EmotionModel(Wav2Vec2PreTrainedModel) snippet
        # breaks under transformers 5.x — load the backbone the standard way
        # (base-model prefix "wav2vec2." is stripped automatically) and load
        # the regression head weights by hand from the safetensors.
        backbone = Wav2Vec2Model.from_pretrained(MODEL)
        cfg = backbone.config
        sd = load_file(hf_hub_download(MODEL, "model.safetensors"))

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)
                self.out_proj = torch.nn.Linear(cfg.hidden_size, 3)  # A, D, V

            def forward(self, x):
                return self.out_proj(torch.tanh(self.dense(x)))

        head = Head()
        head.load_state_dict({k.removeprefix("classifier."): v
                              for k, v in sd.items()
                              if k.startswith("classifier.")})

        class EmotionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone, self.head = backbone, head

            def forward(self, input_values):
                hidden = self.backbone(input_values)[0].mean(dim=1)
                return self.head(hidden)   # (batch, 3) = arousal, dominance, valence

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = EmotionModel().to(device).eval()
        processor = Wav2Vec2Processor.from_pretrained(MODEL)
        _model = (processor, model, device)
    return _model


def _region_tags(pa: float, pv: float, f0_unstable: bool) -> list[tuple[str, float]]:
    """(tag, score) from arousal/valence percentiles within the speaker."""
    tags = []
    if pa >= 0.80 and pv >= 0.60:
        tags.append(("excited", (pa - 0.80) / 0.20 * 0.5 + (pv - 0.60) / 0.40 * 0.5))
    if pa >= 0.80 and pv <= 0.40 and f0_unstable:
        tags.append(("nervous", (pa - 0.80) / 0.20 * 0.5 + (0.40 - pv) / 0.40 * 0.5))
    if 0.60 <= pa and pv <= 0.30:
        tags.append(("frustrated", (pa - 0.60) / 0.40 * 0.5 + (0.30 - pv) / 0.30 * 0.5))
    if pa <= 0.30 and pv <= 0.30:
        tags.append(("sorrowful", (0.30 - pa) / 0.30 * 0.5 + (0.30 - pv) / 0.30 * 0.5))
    if pa <= 0.30 and pv >= 0.60:
        tags.append(("calm", (0.30 - pa) / 0.30 * 0.5 + (pv - 0.60) / 0.40 * 0.5))
    return [(t, round(min(1.0, 0.3 + s), 3)) for t, s in tags]


class StatesAudeering(Detector):
    name = "states"
    tags = ["excited", "nervous", "frustrated", "sorrowful", "calm"]
    source = "original"

    def detect(self, job_dir: Path, segments: list[dict], report) -> list[dict]:
        import soundfile as sf
        import torch

        report("loading audeering A/V model", 0.02)
        processor, model, device = _load()
        pros = features(job_dir, segments, report)

        # pass 1: raw arousal/valence per segment
        av: dict[str, tuple[float, float]] = {}
        for i, seg in enumerate(segments):
            try:
                x, sr = sf.read(job_dir / seg["wav"], dtype="float32")
                inp = processor(x, sampling_rate=sr, return_tensors="pt")
                with torch.inference_mode():
                    y = model(inp.input_values.to(device))[0].cpu().numpy()
                av[seg["seg_id"]] = (float(y[0]), float(y[2]))   # arousal, valence
            except Exception:
                pass
            if (i + 1) % 25 == 0 or i + 1 == len(segments):
                report(f"arousal/valence {i + 1}/{len(segments)}",
                       0.1 + 0.75 * (i + 1) / len(segments))

        # pass 2: per-speaker percentiles -> region map
        by_spk: dict[str, list[dict]] = {}
        for seg in segments:
            by_spk.setdefault(seg["speaker"], []).append(seg)
        out = []
        for spk, segs in by_spk.items():
            if len([s for s in segs if s["seg_id"] in av]) < MIN_SEGS_PER_SPK:
                continue
            a_of = {s["seg_id"]: av[s["seg_id"]][0] for s in segs if s["seg_id"] in av}
            v_of = {s["seg_id"]: av[s["seg_id"]][1] for s in segs if s["seg_id"] in av}
            std_of = {s["seg_id"]: pros.get(s["seg_id"], {}).get("f0_std_st")
                      for s in segs}
            std_vals = [v for v in std_of.values() if v is not None]
            std_p75 = float(np.percentile(std_vals, 75)) if std_vals else 1e9
            for s in segs:
                if s["seg_id"] not in av:
                    continue
                a, v = av[s["seg_id"]]
                pa = speaker_percentile(a_of, a)
                pv = speaker_percentile(v_of, v)
                unstable = (std_of.get(s["seg_id"]) or 0) > std_p75
                for tag, score in _region_tags(pa, pv, unstable):
                    out.append({
                        "seg_id": s["seg_id"], "tag": tag,
                        "score": score,
                        "detector": "audeering_av_percentile",
                        "position_s": None,
                        "evidence": {"arousal": round(a, 3), "valence": round(v, 3),
                                     "pctl_a": round(pa, 3), "pctl_v": round(pv, 3),
                                     "weak": True,   # English SER on Hindi
                                     "device": device},
                    })
        report(f"{len(out)} state candidates", 1.0)
        return out
