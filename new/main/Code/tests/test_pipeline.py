from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from app.config import AnalysisParams
from app.pipeline import SpeechTempoPipeline


def write_test_wav(path: Path, sample_rate: int = 16000) -> None:
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = (0.2 * np.sin(2 * math.pi * 220 * t)).astype(np.float32)
    silence = np.zeros(sample_rate // 2, dtype=np.float32)
    audio = np.concatenate([tone, silence, tone])
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class PipelineTests(unittest.TestCase):
    def test_analyze_without_text(self):
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / "a.wav"
            write_test_wav(audio_path)

            pipeline = SpeechTempoPipeline()
            result = pipeline.analyze(str(audio_path), AnalysisParams())

            self.assertIn("summary", result)
            self.assertIn("tempo_series", result)
            self.assertGreaterEqual(result["summary"]["word_count"], 1)
            self.assertGreaterEqual(len(result["tempo_series"]["time"]), 1)

    def test_analyze_with_text(self):
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / "b.wav"
            write_test_wav(audio_path)

            pipeline = SpeechTempoPipeline()
            result = pipeline.analyze(
                str(audio_path),
                AnalysisParams(),
                reference_text="Это тестовый текст для выравнивания",
            )

            self.assertIsNotNone(result["text_heatmap"])
            self.assertGreaterEqual(len(result["text_heatmap"]["words"]), 1)

    def test_compare(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.wav"
            p2 = Path(td) / "s2.wav"
            write_test_wav(p1)
            write_test_wav(p2)

            pipeline = SpeechTempoPipeline()
            result = pipeline.compare(
                [
                    {"label": "sp1", "audio_path": str(p1)},
                    {"label": "sp2", "audio_path": str(p2)},
                ],
                AnalysisParams(),
            )

            self.assertEqual(len(result["speakers"]), 2)
            self.assertIn("tempo_overlay", result["visuals"])


if __name__ == "__main__":
    unittest.main()
