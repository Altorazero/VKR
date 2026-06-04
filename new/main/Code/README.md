# Speech Tempo Analysis App

Реализация исследовательского приложения для анализа темпа художественной речи.

## Возможности
- Анализ аудио без текста (реальный Silero VAD + Whisper ASR при наличии моделей).
- Анализ по эталонному тексту (реальный WhisperX forced alignment при наличии модели).
- Автоматический fallback на встроенные адаптеры, если ML-модели недоступны.
- Сравнение нескольких дикторов.
- Метрики: WPM, слова/сек, паузы, локальный темп, производные темпа.
- Спектральный анализ: FFT и wavelet.
- Визуализации на Plotly.
- Экспорт в JSON/CSV и текстовый отчёт.

## Базовый запуск
```bash
cd /tmp/workspace/Altorazero/VKR/new/main/Code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AUDIO_ROOT=$(pwd)
uvicorn app.main:app --reload
```

## Подключение реальных моделей
```bash
pip install -r requirements-ml.txt
```

После установки `requirements-ml.txt` пайплайн автоматически использует:
- `silero-vad` для VAD,
- `openai-whisper` для ASR,
- `whisperx` для forced alignment.

## API
- `GET /health`
- `POST /analyze`
- `POST /analyze-with-text`
- `POST /compare`

## Примечание
Для безопасности API обрабатывает только `.wav` файлы внутри директории `AUDIO_ROOT` и ожидает относительные пути.
В ответе `summary.adapters` показывается, какие адаптеры реально были использованы (real/fallback).
