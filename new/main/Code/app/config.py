from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AnalysisMode = Literal["window", "event"]
SmoothingType = Literal["none", "moving_average", "gaussian"]
TempoUnit = Literal["words", "syllables", "phonemes"]


@dataclass(slots=True)
class AnalysisParams:
    vad_threshold: float = 0.02
    min_pause_duration: float = 0.2
    window_size: float = 3.0
    window_step: float = 0.5
    smoothing: SmoothingType = "moving_average"
    tempo_unit: TempoUnit = "words"
    analysis_mode: AnalysisMode = "window"
    wavelet: str = "morl"
    wavelet_scales_min: int = 1
    wavelet_scales_max: int = 64
