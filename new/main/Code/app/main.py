from __future__ import annotations

from fastapi import FastAPI

from .config import AnalysisParams
from .models import AnalyzeRequest, AnalyzeWithTextRequest, AnalysisResponse, CompareRequest, CompareResponse
from .pipeline import SpeechTempoPipeline

app = FastAPI(title="Speech Tempo Analysis", version="0.1.0")
pipeline = SpeechTempoPipeline()


def to_params(model) -> AnalysisParams:
    return AnalysisParams(
        vad_threshold=model.vad_threshold,
        min_pause_duration=model.min_pause_duration,
        window_size=model.window_size,
        window_step=model.window_step,
        smoothing=model.smoothing,
        tempo_unit=model.tempo_unit,
        wavelet=model.wavelet,
        wavelet_scales_min=model.wavelet_scales_min,
        wavelet_scales_max=model.wavelet_scales_max,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalyzeRequest):
    return pipeline.analyze(req.audio_path, to_params(req.params))


@app.post("/analyze-with-text", response_model=AnalysisResponse)
def analyze_with_text(req: AnalyzeWithTextRequest):
    return pipeline.analyze(req.audio_path, to_params(req.params), req.reference_text)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    items = [item.model_dump() for item in req.items]
    return pipeline.compare(items, to_params(req.params))
