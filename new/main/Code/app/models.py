from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalysisParamsModel(BaseModel):
    vad_threshold: float = 0.02
    min_pause_duration: float = 0.2
    window_size: float = 3.0
    window_step: float = 0.5
    smoothing: str = "moving_average"
    tempo_unit: str = "words"
    wavelet: str = "morl"
    wavelet_scales_min: int = 1
    wavelet_scales_max: int = 64


class AnalyzeRequest(BaseModel):
    audio_path: str
    params: AnalysisParamsModel = Field(default_factory=AnalysisParamsModel)


class AnalyzeWithTextRequest(BaseModel):
    audio_path: str
    reference_text: str
    params: AnalysisParamsModel = Field(default_factory=AnalysisParamsModel)


class CompareItem(BaseModel):
    label: str
    audio_path: str
    reference_text: str | None = None


class CompareRequest(BaseModel):
    items: list[CompareItem]
    params: AnalysisParamsModel = Field(default_factory=AnalysisParamsModel)


class AnalysisResponse(BaseModel):
    summary: dict[str, Any]
    tempo_series: dict[str, list[float]]
    derivatives: dict[str, list[float]]
    pauses: dict[str, Any]
    spectrum_fft: dict[str, list[float]]
    wavelet: dict[str, Any]
    text_heatmap: dict[str, Any] | None = None
    visuals: dict[str, Any]


class CompareResponse(BaseModel):
    speakers: list[dict[str, Any]]
    comparison: dict[str, Any]
    visuals: dict[str, Any]
