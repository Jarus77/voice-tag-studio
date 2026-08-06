"""Detector registry — one entry per pluggable tag detector.

Ordered by ear-verifiability (the quality-test-first protocol): acoustic
events first, fuzzy emotional states last. deadpan/playfully/resigned are
deliberately absent — they belong to a future audio-LLM judge tier.
"""

from __future__ import annotations

from .base import Detector
from .beats_rules import BeatsRules
from .laughs_panns import LaughsPanns
from .reactions_panns import ReactionsPanns
from .states_audeering import StatesAudeering
from .tones_prosody import TonesProsody
from .whispers_voicing import WhispersVoicing

DETECTORS: dict[str, Detector] = {
    "laughs": LaughsPanns(),          # PANNs SED — laughter family
    "reactions": ReactionsPanns(),    # PANNs SED — sigh, gasps
    "beats": BeatsRules(),            # rules over alignment — pauses/hesitates/stammers
    "whispers": WhispersVoicing(),    # physics — voicing ratio
    "tones": TonesProsody(),          # per-speaker prosody percentiles — flatly/cheerfully
    "states": StatesAudeering(),      # audeering A/V percentiles — 5 states (weak evidence)
}
