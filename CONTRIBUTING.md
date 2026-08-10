# Contributing

Thanks for looking! This project needs ears as much as code — most open work
is "run a detector on real audio and judge it", which anyone with headphones
can do.

## Setup

Follow the Quick start in the README. `python server.py`, upload any short
Hindi/Hinglish (or other-language!) audio, click **Run remaining ▸**. If the
Preflight banner is green, you're ready.

## Where help is most wanted

1. **Detector bake-offs** — `DETECTOR_EVAL_PLAN.md` is the public roadmap.
   Phases 2–6 (reactions, vocal effort, beats tuning, emotion2vec+ states,
   audio-LLM judge) are designed but unrun. Each phase is a self-contained
   evaluation with a written protocol.
2. **Languages beyond Hindi** — the pipeline is language-agnostic except the
   ASR prompt (`studio/asr_bakeoff.py::GEMINI_PROMPT`) and uroman
   romanization. Try your language, file an issue with what broke.
3. **Gold labeling mode** — a blind-audition UI to measure per-detector
   precision (Phase 0′ in the eval plan). The single highest-leverage missing
   piece.
4. **Speed** — detector-level optimizations are mapped in the issues.

## Rules the code must keep (non-negotiable)

These are load-bearing — PRs that break them won't merge:

- **Text↔audio correspondence is sacred.** A clip's text must describe exactly
  its audio. Boundaries only in verified word gaps; clips of one speaker never
  overlap; silence gets `[pauses]`/`[silence]` tags.
- **Never train on two voices.** Overlapped seconds never enter a clip.
- **Separated/reconstructed audio never trains.** It informs ASR/alignment only.
- **Precision over recall for tags.** A wrong tag is hallucination fuel; a
  missed tag is merely unused data.
- **Keep and flag, don't silently drop.** Anything excluded from the dataset
  stays visible in the UI with its reason.

## PR conventions

- Small, focused PRs. One behavior change each.
- If you touched the segment/export path, paste the before/after counts from a
  re-run (`N clips · M rows · dropped {...}`) in the PR description.
- If you added or changed a detector, include 3–5 auditioned examples
  (timestamp + verdict) — your ear is part of the review.
- No new hard dependencies without discussion (local-first is a feature).

## Reporting bugs

The most valuable bug report here is audio-grounded: job id, clip id, what you
*heard* vs what the text said. Screenshots of the Segments row help.
