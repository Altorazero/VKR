from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .config import AnalysisParams
from .models import AnalyzeRequest, AnalyzeWithTextRequest, AnalysisResponse, CompareRequest, CompareResponse
from .pipeline import SpeechTempoPipeline

app = FastAPI(title="Speech Tempo Analysis", version="0.1.0")
pipeline = SpeechTempoPipeline()

# Добавляем CORS для доступа с фронтенда
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздача статических файлов
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


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


@app.get("/")
def root():
    """Раздача главной страницы с интерфейсом"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Welcome to Speech Tempo Analysis API"}


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
