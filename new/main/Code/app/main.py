from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import shutil
import uuid
import datetime
import io
from sqlalchemy.orm import Session
from pydub import AudioSegment

from .config import AnalysisParams
from .models import AnalyzeRequest, AnalyzeWithTextRequest, AnalysisResponse, CompareRequest, CompareResponse
from .pipeline import SpeechTempoPipeline
from .database import SessionLocal, init_db, AudioRecord, AnalysisResult

# Инициализация БД
init_db()

app = FastAPI(title="Speech Tempo Analysis", version="0.1.0")
pipeline = SpeechTempoPipeline()

# Добавляем CORS для доступа с фронтенда
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздача статических файлов
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Папка для записей
RECORDS_DIR = "records"
if not os.path.exists(RECORDS_DIR):
    os.makedirs(RECORDS_DIR)

app.mount("/records", StaticFiles(directory=RECORDS_DIR), name="records")

# Dependency для получения сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def to_params(model) -> AnalysisParams:
    return AnalysisParams(
        vad_threshold=model.vad_threshold,
        min_pause_duration=model.min_pause_duration,
        window_size=model.window_size,
        window_step=model.window_step,
        smoothing=model.smoothing,
        tempo_unit=model.tempo_unit,
        analysis_mode=model.analysis_mode,
        wavelet=model.wavelet,
        wavelet_scales_min=model.wavelet_scales_min,
        wavelet_scales_max=model.wavelet_scales_max,
    )

@app.get("/")
def root():
    """Раздача главной страницы с интерфейсом"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Welcome to Speech Tempo Analysis API"}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...), 
    author_name: str = Form(None),
    reference_text: str = Form(None),
    db: Session = Depends(get_db)
):
    """Загрузка аудио, конвертация в стандартный WAV и запись в БД"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]
    
    # Считываем данные во временный буфер
    content = await file.read()
    audio_data = io.BytesIO(content)
    
    try:
        # Пытаемся прочитать то, что прислал браузер (может быть webm под видом wav)
        audio = AudioSegment.from_file(audio_data)
        
        # Конвертируем в строго стандартный WAV: 16кГц, моно
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        if author_name:
            safe_name = "".join([c for c in author_name if c.isalnum() or c in (' ', '_')]).rstrip()
            filename = f"{safe_name}_{timestamp}.wav"
        else:
            filename = f"record_{timestamp}_{random_id}.wav"
            
        file_path = os.path.join(RECORDS_DIR, filename)
        
        # Сохраняем уже настоящий WAV
        audio.export(file_path, format="wav")
        
    except Exception as e:
        print(f"Conversion error: {e}")
        # Если pydub не справился, сохраняем как есть
        if author_name:
            safe_name = "".join([c for c in author_name if c.isalnum() or c in (' ', '_')]).rstrip()
            filename = f"{safe_name}_{timestamp}.wav"
        else:
            filename = f"record_{timestamp}_{random_id}.wav"
        file_path = os.path.join(RECORDS_DIR, filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    # Сохранение в БД
    db_record = AudioRecord(
        filename=filename,
        file_path=file_path,
        reference_text=reference_text
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {"id": db_record.id, "filename": filename, "path": filename}

@app.post("/delete-audio")
def delete_audio(filename: str, db: Session = Depends(get_db)):
    """Перемещение файла в папку trash и удаление из активного списка"""
    TRASH_DIR = "trash"
    if not os.path.exists(TRASH_DIR):
        os.makedirs(TRASH_DIR)
        
    src = os.path.join(RECORDS_DIR, filename)
    if os.path.exists(src):
        dst = os.path.join(TRASH_DIR, filename)
        shutil.move(src, dst)
        
    # Удаляем из БД (опционально, или просто помечаем)
    db.query(AudioRecord).filter(AudioRecord.filename == filename).delete()
    db.commit()
    return {"status": "deleted"}

@app.post("/rename-audio")
def rename_audio(old_name: str, new_name: str, db: Session = Depends(get_db)):
    """Переименование аудиофайла"""
    if not new_name.lower().endswith(".wav"):
        new_name += ".wav"
        
    old_path = os.path.join(RECORDS_DIR, old_name)
    new_path = os.path.join(RECORDS_DIR, new_name)
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        
    # Обновляем в БД
    record = db.query(AudioRecord).filter(AudioRecord.filename == old_name).first()
    if record:
        record.filename = new_name
        record.file_path = new_path
        db.commit()
    return {"status": "renamed", "new_name": new_name}

@app.post("/clear-database")
def clear_database(db: Session = Depends(get_db)):
    """Полная очистка таблиц результатов и записей"""
    db.query(AnalysisResult).delete()
    db.query(AudioRecord).delete()
    db.commit()
    return {"status": "database cleared"}

@app.post("/save-ref-text")
def save_ref_text(filename: str, text: str, db: Session = Depends(get_db)):
    """Сохранение эталонного текста без анализа"""
    record = db.query(AudioRecord).filter(AudioRecord.filename == filename).first()
    if record:
        record.reference_text = text
        db.commit()
        return {"status": "saved"}
    raise HTTPException(status_code=404, detail="Record not found")

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    """Получение списка всех файлов в records/ и данных из БД"""
    # 1. Получаем все .wav файлы из папки
    files_in_folder = []
    if os.path.exists(RECORDS_DIR):
        for f in os.listdir(RECORDS_DIR):
            if f.lower().endswith(".wav"):
                files_in_folder.append(f)
                
    # 2. Получаем данные из БД
    db_records = db.query(AudioRecord).all()
    db_map = {r.filename: r for r in db_records}
    
    result = []
    for filename in files_in_folder:
        db_rec = db_map.get(filename)
        
        # Текст берем строго из БД, если записи нет - пустая строка
        ref_text = ""
        if db_rec and db_rec.reference_text:
            ref_text = db_rec.reference_text
            
        result.append({
            "id": db_rec.id if db_rec else None,
            "filename": filename,
            "path": filename, 
            "text": ref_text,
            "date": db_rec.created_at.isoformat() if db_rec else datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(RECORDS_DIR, filename))).isoformat()
        })
        
    # Сортируем по дате (новые сверху)
    result.sort(key=lambda x: x["date"], reverse=True)
    return result

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalyzeRequest, db: Session = Depends(get_db)):
    result = pipeline.analyze(req.audio_path, to_params(req.params))
    
    # Сохраняем результат анализа в БД, если нашли запись аудио
    filename = os.path.basename(req.audio_path)
    audio_rec = db.query(AudioRecord).filter(AudioRecord.filename == filename).first()
    if audio_rec:
        db_result = AnalysisResult(audio_id=audio_rec.id, data=result)
        db.add(db_result)
        db.commit()
        
    return result

@app.post("/analyze-with-text", response_model=AnalysisResponse)
def analyze_with_text(req: AnalyzeWithTextRequest, db: Session = Depends(get_db)):
    result = pipeline.analyze(req.audio_path, to_params(req.params), req.reference_text)
    
    filename = os.path.basename(req.audio_path)
    audio_rec = db.query(AudioRecord).filter(AudioRecord.filename == filename).first()
    if audio_rec:
        # Обновляем эталонный текст в записи аудио
        audio_rec.reference_text = req.reference_text
        
        db_result = AnalysisResult(audio_id=audio_rec.id, data=result)
        db.add(db_result)
        db.commit()
        
    return result

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    items = []
    for item in req.items:
        items.append({
            "label": item.label,
            "audio_path": item.audio_path,
            "reference_text": item.reference_text
        })
    return pipeline.compare(items, to_params(req.params))
