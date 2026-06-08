from __future__ import annotations

from dataclasses import asdict
from typing import Any
import numpy as np

from .adapters import ASRAdapter, AlignerAdapter, VADAdapter
from .config import AnalysisParams
from .io_utils import load_wav_mono, resolve_audio_path
from .metrics import (
    acceleration_stats,
    build_tempo_series,
    derivatives,
    fft_spectrum,
    pauses_from_speech,
    smooth,
    summarize_pauses,
    wavelet_transform,
)
from .visualization import compare_figure, spectrum_figure, tempo_figure, text_heatmap, wavelet_figure


class SpeechTempoPipeline:
    def __init__(self) -> None:
        self.vad = VADAdapter()
        self.asr = ASRAdapter()
        self.aligner = AlignerAdapter()

    def analyze(self, audio_path: str, params: AnalysisParams, reference_text: str | None = None) -> dict[str, Any]:
        resolved_audio_path = resolve_audio_path(audio_path)
        audio, sample_rate = load_wav_mono(resolved_audio_path)
        total_duration = len(audio) / sample_rate

        speech = self.vad.detect(audio, sample_rate, params.vad_threshold)
        pauses = pauses_from_speech(speech, total_duration, params.min_pause_duration)

        words = (
            self.aligner.align(reference_text, resolved_audio_path, audio, sample_rate, speech)
            if reference_text
            else self.asr.transcribe(resolved_audio_path, audio, sample_rate, speech)
        )

        # Анализ для 4-х разных режимов
        modes = [
            ("window", "words", "WPM (Window)"),
            ("event", "words", "WPM (Words)"),
            ("event", "syllables", "Syllables/s"),
            ("event", "phonemes", "Letters/s"),
        ]
        
        all_series = {}
        all_visuals = {}

        for mode_name, unit, y_label in modes:
            time, tempo, labels = build_tempo_series(
                words=words,
                total_duration=total_duration,
                unit=unit,
                window_size=params.window_size,
                step=0.5, # Фиксированный шаг для консистентности
                mode=mode_name,
            )
            tempo_smoothed = smooth(tempo, params.smoothing)
            d1, d2 = derivatives(time, tempo_smoothed)
            
            key = f"{mode_name}_{unit}"
            all_series[key] = {
                "time": time.tolist(),
                "tempo": tempo_smoothed.tolist(),
                "d1": d1.tolist(),
                "labels": labels
            }
            
            all_visuals[f"timeline_{key}"] = tempo_figure(
                time.tolist(),
                tempo_smoothed.tolist(),
                d1.tolist(),
                [[a, b] for a, b in speech],
                [[a, b] for a, b in pauses],
                labels=labels if mode_name == "event" else None,
                y_label=y_label
            )

        # Используем первый режим (window_words) для FFT и Wavelet по умолчанию
        main_series = all_series["window_words"]
        freqs, mag = fft_spectrum(np.array(main_series["time"]), np.array(main_series["tempo"]))
        scales, power = wavelet_transform(
            np.array(main_series["tempo"]),
            wavelet=params.wavelet,
            min_scale=params.wavelet_scales_min,
            max_scale=params.wavelet_scales_max,
        )

        pause_summary = summarize_pauses(pauses, total_duration)
        speech_duration = sum(e - s for s, e in speech)
        word_count = len(words)

        summary = {
            "word_count": word_count,
            "speech_duration": speech_duration,
            "pause_duration": pause_summary["total_pause_duration"],
            "speech_to_pause_ratio": float(speech_duration / max(pause_summary["total_pause_duration"], 1e-9)),
            "wpm": float(word_count / max(speech_duration, 1e-9) * 60.0),
            "words_per_second": float(word_count / max(speech_duration, 1e-9)),
            "params": asdict(params),
            "acceleration_stats": acceleration_stats(np.array(main_series["d1"])),
            "adapters": {
                "vad": self.vad.backend,
                "asr": self.asr.backend,
                "aligner": self.aligner.backend,
            },
        }

        heatmap = None
        if reference_text:
            word_values = []
            for w in words:
                dur = max(1e-6, w.end - w.start)
                word_values.append(float(60.0 / dur))
            heatmap = {
                "words": [w.word for w in words],
                "values": word_values,
                "figure": text_heatmap([w.word for w in words], word_values),
            }

        all_visuals.update({
            "fft": spectrum_figure(freqs.tolist(), mag.tolist()),
            "wavelet": wavelet_figure(scales.tolist(), power.tolist(), main_series["time"]),
        })

        return {
            "summary": summary,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in words],
            "all_series": all_series,
            "pauses": {
                "segments": [[a, b] for a, b in pauses],
                "stats": pause_summary,
            },
            "spectrum_fft": {"frequency": freqs.tolist(), "magnitude": mag.tolist()},
            "wavelet": {"scales": scales.tolist(), "power": power.tolist()},
            "text_heatmap": heatmap,
            "visuals": all_visuals,
        }

    def compare(self, items: list[dict[str, Any]], params: AnalysisParams) -> dict[str, Any]:
        analyzed = []
        series = []
        for item in items:
            result = self.analyze(item["audio_path"], params, item.get("reference_text"))
            analyzed.append(
                {
                    "label": item["label"],
                    "summary": result["summary"],
                    "pause_stats": result["pauses"]["stats"],
                }
            )
            # Для сравнения используем режим оконных слов
            main_s = result["all_series"]["window_words"]
            series.append(
                {
                    "label": item["label"],
                    "time": main_s["time"],
                    "tempo": main_s["tempo"],
                }
            )

        comparison = {
            "average_wpm": {item["label"]: item["summary"]["wpm"] for item in analyzed},
            "pause_count": {item["label"]: item["pause_stats"]["count"] for item in analyzed},
        }

        return {
            "speakers": analyzed,
            "comparison": comparison,
            "visuals": {"tempo_overlay": compare_figure(series)},
        }
