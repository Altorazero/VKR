# Speech Tempo Analysis App

Реализация исследовательского приложения для анализа темпа художественной речи.

## Возможности
- Анализ аудио без текста (VAD + оценка темпа).
- Анализ по эталонному тексту (выравнивание слов по времени в fallback-режиме).
- Сравнение нескольких дикторов.
- Метрики: WPM, слова/сек, паузы, локальный темп, производные темпа.
- Спектральный анализ: FFT и wavelet.
- Визуализации на Plotly.
- Экспорт в JSON/CSV и текстовый PDF-отчёт.

## Запуск
```bash
cd /tmp/workspace/Altorazero/VKR/new/main/Code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AUDIO_ROOT=/tmp/workspace/Altorazero/VKR/new/main/Code
uvicorn app.main:app --reload
```

## API
- `GET /health`
- `POST /analyze`
- `POST /analyze-with-text`
- `POST /compare`

## Примечание
В текущей версии используется fallback-пайплайн. Архитектура подготовлена для замены адаптеров на Silero VAD / Whisper / WhisperX.
Для безопасности API обрабатывает только `.wav` файлы внутри директории `AUDIO_ROOT`.
