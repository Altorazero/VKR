from __future__ import annotations

from dataclasses import dataclass

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
    def transcribe(self, speech_segments: list[tuple[float, float]]) -> list[WordTiming]:
        total_speech = sum(end - start for start, end in speech_segments)
        estimated_words = max(1, int(total_speech * 2.4))
        return _synthesize_words([f"w{i+1}" for i in range(estimated_words)], speech_segments)


class FallbackAligner:
    def align(self, reference_text: str, speech_segments: list[tuple[float, float]]) -> list[WordTiming]:
        words = [w for w in reference_text.replace("\n", " ").split(" ") if w.strip()]
        if not words:
            return []
        return _synthesize_words(words, speech_segments)


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
