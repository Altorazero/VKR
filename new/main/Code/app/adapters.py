from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass(slots=True)
class WordTiming:
    word: str
    start: float
    end: float


class FallbackVAD:
    def detect(self, audio: np.ndarray, sample_rate: int, threshold: float) -> list[tuple[float, float]]:
        frame_size = int(0.03 * sample_rate)
        hop_size = int(0.01 * sample_rate)
        if frame_size <= 0 or len(audio) < frame_size:
            return []

        energies = []
        times = []
        for i in range(0, len(audio) - frame_size + 1, hop_size):
            frame = audio[i : i + frame_size]
            energy = float(np.sqrt(np.mean(frame * frame)))
            energies.append(energy)
            times.append(i / sample_rate)

        voiced = np.array(energies) > threshold
        segments: list[tuple[float, float]] = []
        start_idx = None
        for idx, is_voiced in enumerate(voiced):
            if is_voiced and start_idx is None:
                start_idx = idx
            elif not is_voiced and start_idx is not None:
                start_time = times[start_idx]
                end_time = times[idx] + frame_size / sample_rate
                segments.append((start_time, end_time))
                start_idx = None

        if start_idx is not None:
            segments.append((times[start_idx], len(audio) / sample_rate))

        return _merge_segments(segments, max_gap=0.08)


class FallbackASR:
    def transcribe(
        self,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> list[WordTiming]:
        del audio_path, audio, sample_rate
        total_speech = sum(end - start for start, end in speech_segments)
        estimated_words = max(1, int(total_speech * 2.4))
        return _synthesize_words([f"w{i + 1}" for i in range(estimated_words)], speech_segments)


class FallbackAligner:
    def align(
        self,
        reference_text: str,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> list[WordTiming]:
        del audio_path, audio, sample_rate
        words = [w for w in reference_text.replace("\n", " ").split(" ") if w.strip()]
        if not words:
            return []
        return _synthesize_words(words, speech_segments)


class RealSileroVAD:
    def __init__(self) -> None:
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        try:
            import silero_vad  # noqa: F401

            return True
        except Exception:
            return False

    def detect(self, audio: np.ndarray, sample_rate: int, threshold: float) -> list[tuple[float, float]]:
        if not self.available:
            raise RuntimeError("silero_vad is not available")
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        if self._model is None:
            self._model = load_silero_vad()

        waveform = torch.tensor(audio, dtype=torch.float32)
        speech_timestamps = get_speech_timestamps(
            waveform,
            self._model,
            sampling_rate=sample_rate,
            threshold=threshold,
            return_seconds=False,
        )
        segments = [
            (ts["start"] / sample_rate, ts["end"] / sample_rate)
            for ts in speech_timestamps
        ]
        return _merge_segments(segments, max_gap=0.05)


class RealWhisperASR:
    def __init__(self, model_size: str = "base") -> None:
        self.model_size = model_size
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        try:
            import whisper  # noqa: F401

            return True
        except Exception:
            return False

    def transcribe(
        self,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> list[WordTiming]:
        del audio_path
        if not self.available:
            raise RuntimeError("whisper is not available")
        import whisper

        if self._model is None:
            self._model = whisper.load_model(self.model_size)

        resampled = _resample_if_needed(audio, sample_rate, target_rate=16000)
        result = self._model.transcribe(resampled, fp16=False, word_timestamps=True)
        words = _extract_whisper_words(result)
        if words:
            return words

        text = str(result.get("text", "")).strip()
        if not text:
            return []
        plain_words = [w for w in re.split(r"\s+", text) if w]
        return _synthesize_words(plain_words, speech_segments)


class RealWhisperXAligner:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._align_models: dict[str, tuple[Any, dict[str, Any]]] = {}

    @property
    def available(self) -> bool:
        try:
            import whisperx  # noqa: F401

            return True
        except Exception:
            return False

    def align(
        self,
        reference_text: str,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> list[WordTiming]:
        del audio_path, speech_segments
        if not self.available:
            raise RuntimeError("whisperx is not available")
        import whisperx

        duration = len(audio) / max(sample_rate, 1)
        if duration <= 0:
            return []

        language_code = _detect_language_code(reference_text)
        if language_code not in self._align_models:
            model, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            self._align_models[language_code] = (model, metadata)

        model, metadata = self._align_models[language_code]
        aligned = whisperx.align(
            transcript=[{"start": 0.0, "end": float(duration), "text": reference_text}],
            model=model,
            align_model_metadata=metadata,
            audio=_resample_if_needed(audio, sample_rate, target_rate=16000),
            device=self.device,
            return_char_alignments=False,
        )
        word_segments = aligned.get("word_segments", [])
        words: list[WordTiming] = []
        for segment in word_segments:
            word = str(segment.get("word", "")).strip()
            start = segment.get("start")
            end = segment.get("end")
            if not word or start is None or end is None:
                continue
            words.append(WordTiming(word=word, start=float(start), end=float(end)))
        return words


class VADAdapter:
    def __init__(self) -> None:
        self.real = RealSileroVAD()
        self.fallback = FallbackVAD()
        self.backend = "silero_vad" if self.real.available else "fallback_vad"

    def detect(self, audio: np.ndarray, sample_rate: int, threshold: float) -> list[tuple[float, float]]:
        if self.real.available:
            try:
                self.backend = "silero_vad"
                return self.real.detect(audio, sample_rate, threshold)
            except Exception:
                self.backend = "fallback_vad"
        return self.fallback.detect(audio, sample_rate, threshold)


class ASRAdapter:
    def __init__(self) -> None:
        self.real = RealWhisperASR()
        self.fallback = FallbackASR()
        self.backend = "whisper" if self.real.available else "fallback_asr"

    def transcribe(
        self,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> list[WordTiming]:
        if self.real.available:
            try:
                self.backend = "whisper"
                return self.real.transcribe(audio_path, audio, sample_rate, speech_segments)
            except Exception:
                self.backend = "fallback_asr"
        return self.fallback.transcribe(audio_path, audio, sample_rate, speech_segments)


class AlignerAdapter:
    def __init__(self) -> None:
        self.real = RealWhisperXAligner()
        self.fallback = FallbackAligner()
        self.backend = "whisperx" if self.real.available else "fallback_aligner"

    def align(
        self,
        reference_text: str,
        audio_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> list[WordTiming]:
        if self.real.available:
            try:
                self.backend = "whisperx"
                words = self.real.align(reference_text, audio_path, audio, sample_rate, speech_segments)
                if words:
                    return words
                self.backend = "fallback_aligner"
            except Exception:
                self.backend = "fallback_aligner"
        return self.fallback.align(reference_text, audio_path, audio, sample_rate, speech_segments)


def _extract_whisper_words(result: dict[str, Any]) -> list[WordTiming]:
    words: list[WordTiming] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word = str(w.get("word", "")).strip()
            start = w.get("start")
            end = w.get("end")
            if not word or start is None or end is None:
                continue
            words.append(WordTiming(word=word, start=float(start), end=float(end)))
    return words


def _resample_if_needed(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32)
    from scipy import signal

    gcd = np.gcd(sample_rate, target_rate)
    up = target_rate // gcd
    down = sample_rate // gcd
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def _detect_language_code(text: str) -> str:
    has_cyrillic = bool(re.search(r"[\u0400-\u04FF]", text))
    return "ru" if has_cyrillic else "en"


def _synthesize_words(words: list[str], segments: list[tuple[float, float]]) -> list[WordTiming]:
    if not segments:
        return []
    total = sum(max(0.0, e - s) for s, e in segments)
    if total <= 0:
        return []

    out: list[WordTiming] = []
    idx = 0
    for seg_start, seg_end in segments:
        seg_duration = seg_end - seg_start
        if seg_duration <= 0:
            continue
        seg_share = seg_duration / total
        count = max(1, int(round(len(words) * seg_share)))
        chunk = words[idx : idx + count]
        idx += count
        if not chunk:
            continue
        word_len = seg_duration / len(chunk)
        for j, word in enumerate(chunk):
            ws = seg_start + j * word_len
            we = min(seg_end, ws + word_len)
            out.append(WordTiming(word=word, start=ws, end=we))

    while idx < len(words) and out:
        last = out[-1]
        out.append(WordTiming(word=words[idx], start=last.end, end=last.end + 0.05))
        idx += 1

    return out


def _merge_segments(segments: list[tuple[float, float]], max_gap: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    segments = sorted(segments)
    merged = [segments[0]]
    for start, end in segments[1:]:
        m_start, m_end = merged[-1]
        if start - m_end <= max_gap:
            merged[-1] = (m_start, max(m_end, end))
        else:
            merged.append((start, end))
    return merged
