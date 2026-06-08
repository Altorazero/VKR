from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import soundfile as sf

def resolve_audio_path(audio_path: str) -> Path:
    # 1. Если это уже существующий абсолютный путь
    path = Path(audio_path)
    if path.is_absolute() and path.is_file():
        return path
        
    root = Path(os.getcwd()).resolve()
    
    # 2. Проверяем в папке records/
    filename = os.path.basename(audio_path)
    records_path = root / "records" / filename
    if records_path.is_file():
        return records_path
        
    # 3. Проверяем просто в корне
    local_path = root / filename
    if local_path.is_file():
        return local_path

    raise FileNotFoundError(f"Audio file not found: {audio_path}")

def load_wav_mono(audio_path: str | Path) -> tuple[np.ndarray, int]:
    path = audio_path if isinstance(audio_path, Path) else resolve_audio_path(audio_path)
    
    # soundfile читает почти все форматы и сразу возвращает numpy
    data, sample_rate = sf.read(str(path))
    
    # Конвертация в моно, если стерео
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    return data.astype(np.float32), sample_rate
