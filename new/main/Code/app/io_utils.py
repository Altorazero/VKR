from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np


def load_wav_mono(audio_path: str) -> tuple[np.ndarray, int]:
    root = Path(os.getenv("AUDIO_ROOT", os.getcwd())).resolve()
    path = Path(audio_path).expanduser().resolve(strict=True)

    if path.suffix.lower() != ".wav":
        raise ValueError("Only .wav files are supported")
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if root not in path.parents and path != root:
        raise ValueError(f"Audio path must be inside AUDIO_ROOT: {root}")

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
