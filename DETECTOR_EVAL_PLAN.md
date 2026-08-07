# Detector & ASR evaluation plan (agreed 2026-08-07)

Decisions baked in: research/internal use for now (NC-licensed models allowed,
re-audit before any commercial ship) · cloud APIs allowed (Sarvam, Gemini) ·
gold set deferred to Phase 0' (between 5 and 6) — phases 1–5 are QUALITATIVE
side-by-side auditions on short uploaded clips in the workbench; the gold set
then quantifies the finalists. Nothing below is implemented until Suraj says
"go" on that phase.

## Order

### Phase 1 — ASR bake-off  ← FIRST (known pain: srota misses/garbles)
- Candidates: srota (baseline) · Sarvam saaras codemix · Whisper-large-v3
  (local) · Gemini audio verbatim-prompted · (optional) AI4Bharat IndicConformer.
- Workbench test: same uploaded clip transcribed by each → side-by-side
  transcript view; listen while reading.
- Judge by: misses/truncations (the complaint) · Hinglish script convention ·
  FILLER FIDELITY (मतलब/अं/हाँ preserved? — `hesitates` depends on it; Sarvam
  known to drop fillers).
- Output: new default engine + escalation rule (align-flagged → Gemini).

### Phase 2 — Reactions (laughs / sigh / cries)
- Candidates: PANNs (baseline — measured clean on our audio, don't dismiss) ·
  CED (xiaomi, AudioSet head) · CED/BEATs + VocalSound-finetuned head (needs a
  small linear-probe training job) · Gemini zero-shot.
- Caveats accepted: VocalSound lacks gasp/gulp/cry; AudioSet crying is
  infant-dominated → cries likely weakest tag; gulps stays dropped.

### Phase 3 — Vocal effort (whispers + shouts as ONE continuum)
- Candidates: current heuristics (baseline) · spectral tilt + HNR + H1-H2
  feature gate · WavLM-based speaker-relative effort classifier.
- Watch: Hindi aspirated stops false-triggering the pyin voicing heuristic.

### Phase 4 — Beats
- `pauses`: deterministic alignment — expected keeper.
- `hesitates`: decision follows Phase 1 filler results (deleted filler ==
  silence downstream).
- `stammers`: honest test; likely audio-LLM territory or drop for v1.

### Phase 5 — Emotional states
- Candidates: audeering A/V percentile (baseline; CC-BY-NC — research only,
  BLOCKED for commercial) · emotion2vec+ Large · Gemini captioning → category
  mapping.
- Keep a license column for every candidate from here on.

### Phase 0' — Gold micro-benchmark  (BETWEEN 5 AND 6, by Suraj's call)
- Labeling UI in the workbench: ✓/✗/? buttons on the Tags table →
  gold_labels.jsonl. Suraj labels; pilot 25–50 clips/tag + 50 neutrals for the
  FINALISTS of phases 1–5 only; 30 hand-corrected verbatim transcripts as ASR
  gold. Per-tag precision/recall harness.

### Phase 6 — The collapse experiment
- One audio-LLM (Gemini) prompted for ALL tag families per clip, scored
  against the gold set vs the phase-1–5 finalists. If it wins → pseudo-labeler
  + distill a small local head (the stage-6 judge tier, evidence-based).

## Standing corrections from this review
- audeering MSP-Dim is ENGLISH-trained (never Hindi) and CC-BY-NC.
- General AudioSet taggers underperform on human vocal sounds (VocalSound
  finding) — the reason Phase 2 exists.
- "SOTA" claim for beats applies to `pauses` only.
