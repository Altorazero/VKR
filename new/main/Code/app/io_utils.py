from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np


def load_wav_mono(audio_path: str) -> tuple[np.ndarray, int]:
    root = Path(os.getenv("AUDIO_ROOT", os.getcwd())).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AUDIO_ROOT does not exist: {root}")

    catalog = _audio_catalog(root)
    path = catalog.get(audio_path)
    if path is None:
        raise FileNotFoundError(
            "Audio file not found in AUDIO_ROOT catalog. "
            "Use a relative .wav path from AUDIO_ROOT."
        )

    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        sample_width = wf.getsampwidth()

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def _audio_catalog(root: Path) -> dict[str, Path]:
    catalog: dict[str, Path] = {}
    for wav in root.rglob("*.wav"):
        if not wav.is_file():
            continue
        rel = wav.relative_to(root).as_posix()
        catalog[rel] = wav
    return catalog
