from __future__ import annotations

from typing import Literal

import numpy as np

from .adapters import WordTiming

LATIN_VOWELS = "aeiouy"
CYRILLIC_VOWELS = "аеёиоуыэюя"
SUPPORTED_VOWELS = LATIN_VOWELS + CYRILLIC_VOWELS


def pauses_from_speech(speech_segments: list[tuple[float, float]], total_duration: float, min_pause_duration: float) -> list[tuple[float, float]]:
    pauses: list[tuple[float, float]] = []
    current = 0.0
    for start, end in sorted(speech_segments):
        if start - current >= min_pause_duration:
            pauses.append((current, start))
        current = max(current, end)
    if total_duration - current >= min_pause_duration:
        pauses.append((current, total_duration))
    return pauses


def summarize_pauses(pauses: list[tuple[float, float]], total_duration: float) -> dict:
    durations = np.array([end - start for start, end in pauses], dtype=np.float64)
    if durations.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "frequency_per_min": 0.0,
            "distribution": [],
            "total_pause_duration": 0.0,
        }

    return {
        "count": int(durations.size),
        "mean": float(durations.mean()),
        "median": float(np.median(durations)),
        "frequency_per_min": float(durations.size / max(1e-9, total_duration / 60.0)),
        "distribution": durations.round(4).tolist(),
        "total_pause_duration": float(durations.sum()),
    }


def build_tempo_series(
    words: list[WordTiming],
    total_duration: float,
    unit: Literal["words", "syllables", "phonemes"],
    window_size: float,
    step: float,
    mode: Literal["window", "event"] = "window",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if total_duration <= 0 or not words:
        return np.array([]), np.array([]), []

    if mode == "event":
        times = []
        values = []
        labels = []
        for w in words:
            units = _word_units(w.word, unit)
            duration = max(0.01, w.end - w.start)
            
            if units > 1 and unit != "words":
                # Генерируем суб-метки в зависимости от типа (слоги или фонемы/буквы)
                if unit == "syllables":
                    sub_labels = _split_to_syllables(w.word)
                else: # phonemes (буквы)
                    sub_labels = [c for c in w.word if c.isalpha()]
                
                unit_duration = duration / units
                for j in range(1, units + 1):
                    sub_end = w.start + j * unit_duration
                    tempo = 1.0 / unit_duration
                    times.append(sub_end)
                    values.append(tempo)
                    
                    if j-1 < len(sub_labels):
                        labels.append(sub_labels[j-1])
                    else:
                        labels.append(f".") # Короткая точка вместо повтора слова
            else:
                tempo = units / duration
                if unit == "words":
                    tempo *= 60.0
                times.append(w.end)
                values.append(tempo)
                labels.append(w.word)
        return np.array(times), np.array(values), labels

    # Режим окна (labels не используются)
    centers = np.arange(window_size / 2.0, total_duration + 1e-9, step)
    values = np.zeros_like(centers)

    for i, center in enumerate(centers):
        left = center - window_size / 2.0
        right = center + window_size / 2.0
        selected = [w for w in words if w.start >= left and w.end <= right]
        units = sum(_word_units(w.word, unit) for w in selected)
        values[i] = units / max(window_size, 1e-9)

    if unit == "words":
        values = values * 60.0

    return centers, values, []


def _split_to_syllables(word: str) -> list[str]:
    """Разбивает слово на слоги по гласным."""
    if not word:
        return []
    res = []
    current = ""
    for char in word:
        current += char
        if char.lower() in SUPPORTED_VOWELS:
            res.append(current)
            current = ""
    if current:
        if res:
            res[-1] += current
        else:
            res.append(current)
    return res


def smooth(values: np.ndarray, mode: str) -> np.ndarray:
    if values.size == 0 or mode == "none":
        return values
    if mode == "moving_average":
        if values.size < 5:
            return values
        kernel = np.ones(5, dtype=np.float64) / 5.0
        return np.convolve(values, kernel, mode="same")
    if mode == "gaussian":
        try:
            from scipy.ndimage import gaussian_filter1d

            return gaussian_filter1d(values, sigma=1.0)
        except Exception:
            return values
    return values


def derivatives(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.zeros_like(values), np.zeros_like(values)
    first = np.gradient(values, times)
    second = np.gradient(first, times)
    return first, second


def acceleration_stats(first_derivative: np.ndarray) -> dict:
    if first_derivative.size == 0:
        return {"mean": 0.0, "std": 0.0, "accelerations": 0, "decelerations": 0}
    return {
        "mean": float(first_derivative.mean()),
        "std": float(first_derivative.std()),
        "accelerations": int(np.sum(first_derivative > 0)),
        "decelerations": int(np.sum(first_derivative < 0)),
    }


def fft_spectrum(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 2:
        return np.array([]), np.array([])
    dt = float(np.mean(np.diff(times)))
    centered = values - values.mean()
    spec = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    magnitude = np.abs(spec)
    return freqs, magnitude


def wavelet_transform(values: np.ndarray, wavelet: str, min_scale: int, max_scale: int) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([[]])
    scale_start = max(1, min_scale)
    scale_end = max(scale_start + 1, max_scale + 1)
    scales = np.arange(scale_start, scale_end)
    try:
        import pywt

        coeffs, _freqs = pywt.cwt(values, scales, wavelet)
        power = np.abs(coeffs)
        return scales, power
    except Exception:
        power = np.tile(np.abs(values), (scales.size, 1))
        return scales, power


def _word_units(word: str, unit: str) -> int:
    if unit == "words":
        return 1
    syllables = sum(1 for ch in word.lower() if ch in SUPPORTED_VOWELS)
    if unit == "syllables":
        return max(1, syllables)
    return max(1, len([ch for ch in word if ch.isalpha()]))
