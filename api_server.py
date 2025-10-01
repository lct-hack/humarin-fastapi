# api_server.py

"""
REST API для анализа КТГ (кардиотокография) по загруженным пользователем файлам.
Нет доступа к серверным папкам — только UploadFile (2 CSV: ЧСС и матка).
Есть потоковый (SSE) и мгновенный анализ.
CORS максимально открыт.
"""

import os
import asyncio
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from processing import classify_and_plot_fhr, analyze_and_plot_toco_demo

# ————————————————————————————————————————————————————————————————
# Константы, глобальные структуры
# ————————————————————————————————————————————————————————————————

FS_DEFAULT = 7.87  # частота дискретизации по умолчанию
active_alerts: Dict[str, Dict[str, str]] = {}  # monitor_id -> {alert_key: warning_id}

# ————————————————————————————————————————————————————————————————
# Инициализация FastAPI + CORS (максимально разрешено)
# ————————————————————————————————————————————————————————————————

app = FastAPI(
    title="КТГ Мониторинг API (upload-only)",
    description="Анализ КТГ по двум CSV, загруженным клиентом (ЧСС и матка).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # максимально открыто
    allow_credentials=False,    # если True, нельзя использовать '*'
    allow_methods=["*"],
    allow_headers=["*"],
)

# ————————————————————————————————————————————————————————————————
# Модели данных
# ————————————————————————————————————————————————————————————————

class IntervalPoint(BaseModel):
    # Идентификация и значения
    monitor_id: str = Field(..., description="Уникальный ID монитора")
    time_sec: float = Field(..., description="Время в секундах от начала записи")
    fhr_bpm: Optional[float] = Field(None, description="ЧСС плода, уд/мин")
    uterus_mmhg: Optional[float] = Field(None, description="Давление в матке, мм рт.ст.")
    # Предупреждения
    warning_text: Optional[str] = Field(None, description="Текст предупреждения, если есть")
    warning_id: Optional[str] = Field(None, description="ID предупреждения (постоянный для пары begin/end)")
    warning_status: Optional[str] = Field(None, description="Статус предупреждения: begin/end")
    # Разметка показателей (только коды, без цветов)
    # ЧСС — линия
    fhr_line_status: int = Field(..., description="Статус линии ЧСС: тяж. бради=-2, умер. бради=-1, норма=0, умер. тахи=1, тяж. тахи=2")
    # ЧСС — события (фон на графике ЧСС)
    fhr_event_status: int = Field(..., description="Акселерации/децелерации: 1/-1, 0 если нет")
    # Токограмма — линия (интенсивность)
    toco_line_status: int = Field(..., description="Интенсивность токограммы: 0 для <30 мм рт.ст., 1 для 30–80, 2 для >80")
    # Токограмма — фоны (окна тревог)
    toco_tachysystole: int = Field(..., description="Тахистолия (>5/10 мин): 0/1")
    toco_hypertonus: int = Field(..., description="Гипертонус (tone_med > 25): 0/1")
    toco_tetanic: int = Field(..., description="Тетаническая (≥120 с): 0/1")


class StreamBatch(BaseModel):
    monitor_id: str
    window_start_sec: float
    window_end_sec: float
    points: List[IntervalPoint]


class InstantAnalysisResult(BaseModel):
    monitor_id: str = Field(..., description="ID монитора")
    duration_sec: float = Field(..., description="Длительность записи, сек")
    timeseries: List[IntervalPoint] = Field(..., description="Полный временной ряд с разметкой")
    warnings: List[Dict] = Field(..., description="Список всех предупреждений (begin/end)")

# ————————————————————————————————————————————————————————————————
# Помощники: загрузка CSV, анализ, разметка
# ————————————————————————————————————————————————————————————————

def _read_csv_to_df(upload: UploadFile, fs: float) -> pd.DataFrame:
    """
    Читает CSV из UploadFile. Ожидаемые столбцы:
      - обязательно 'value' (значения)
      - опционально 'time_sec' (если нет — генерируем равномерно от 0 с шагом 1/fs)
    Допускается CSV с одним столбцом — будет считаться 'value'.
    """
    try:
        df = pd.read_csv(upload.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV '{upload.filename}': {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail=f"Пустой файл: {upload.filename}")

    # Определяем столбец 'value'
    if "value" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "value"})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Файл '{upload.filename}' должен содержать столбец 'value' или один столбец со значениями.",
            )

    # Преобразуем к float
    try:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    except Exception:
        raise HTTPException(status_code=400, detail=f"Столбец 'value' в '{upload.filename}' не числовой.")

    if df["value"].isna().all():
        raise HTTPException(status_code=400, detail=f"В '{upload.filename}' нет валидных числовых значений в 'value'.")

    # time_sec
    if "time_sec" in df.columns:
        try:
            df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Столбец 'time_sec' в '{upload.filename}' не числовой.")
        # Если есть пропуски — восстановим по индексу с шагом 1/fs
        if df["time_sec"].isna().any():
            n = len(df)
            df["time_sec"] = np.arange(n) / float(fs)
    else:
        n = len(df)
        df["time_sec"] = np.arange(n) / float(fs)

    # Сортировка по времени и очистка nan
    df = df.dropna(subset=["value", "time_sec"]).sort_values("time_sec").reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"После очистки в '{upload.filename}' не осталось данных.")
    return df[["time_sec", "value"]]


def analyze_data(bpm_df: pd.DataFrame, uterus_df: pd.DataFrame, fs: float):
    min_len = min(len(bpm_df), len(uterus_df))
    if min_len < 10:
        raise HTTPException(status_code=400, detail="Слишком мало данных для анализа")

    bpm_df = bpm_df.iloc[:min_len].reset_index(drop=True)
    uterus_df = uterus_df.iloc[:min_len].reset_index(drop=True)

    fhr_bpm = bpm_df["value"].values
    toco_mmhg = uterus_df["value"].values
    time_arr = bpm_df["time_sec"].values

    # анализ ЧСС
    states, baseline_event, fhr_stats = classify_and_plot_fhr(
        fhr_bpm, fs=fs, class_name="", show=False
    )

    # ⚡️ выравнивание длин
    min_len2 = min(len(fhr_bpm), len(states), len(baseline_event), len(time_arr))
    fhr_bpm = fhr_bpm[:min_len2]
    states = states[:min_len2]
    baseline_event = baseline_event[:min_len2]
    time_arr = time_arr[:min_len2]
    toco_mmhg = toco_mmhg[:min_len2]  # тоже подрежем матку

    # анализ матки
    try:
        tone, intensity, alert_code, contractions, windows_alerts, toco_stats = analyze_and_plot_toco_demo(
            toco_mmhg, fs=fs, show=False
        )
    except Exception as e:
        tone = np.zeros_like(toco_mmhg)
        intensity = toco_mmhg
        alert_code = np.zeros_like(toco_mmhg, dtype=np.int8)
        contractions = []
        windows_alerts = []
        toco_stats = {"error": str(e)}

    return {
        "fhr": {"data": fhr_bpm, "states": states, "baseline": baseline_event, "stats": fhr_stats},
        "toco": {"data": toco_mmhg, "tone": tone, "intensity": intensity, "alert_code": alert_code, "stats": toco_stats},
        "time": time_arr,
        "fs": fs,
    }


# ————————————————————————————————————————————————————————————————
# Разметка статусов + генерация предупреждений
# ————————————————————————————————————————————————————————————————

FHR_STATE_MAP = {  # из processing: 1..5 -> требуемые −2..2
    1: 0,   # Норма
    2: -1,  # Умеренная брадикардия
    3: -2,  # Тяжелая брадикардия
    4: 1,   # Умеренная тахикардия
    5: 2,   # Тяжелая тахикардия
}
FHR_STATE_LABELS = {
    -2: "Тяжелая брадикардия",
    -1: "Умеренная брадикардия",
     0: "Норма",
     1: "Умеренная тахикардия",
     2: "Тяжелая тахикардия",
}
ALERT_LABELS = {1: "Тахистолия", 2: "Гипертонус", 3: "Тетаническая"}


def fhr_event_status(delta: float, enter: float = 12.0) -> int:
    """Акселерации/децелерации по отклонению от локальной базы (baseline_event). 1/-1/0."""
    if delta >= enter:
        return 1
    if delta <= -enter:
        return -1
    return 0


def toco_line_band(intensity_mmHg: float) -> int:
    """Интенсивность токограммы: 0 (<30), 1 (30–80), 2 (>80)."""
    if intensity_mmHg > 80.0:
        return 2
    if intensity_mmHg >= 30.0:
        return 1
    return 0


def begin_alert(monitor_id: str, key: str) -> Tuple[str, str]:
    m = active_alerts.setdefault(monitor_id, {})
    wid = str(uuid.uuid4())
    m[key] = wid
    return wid, "begin"


def end_alert(monitor_id: str, key: str) -> Tuple[Optional[str], Optional[str]]:
    m = active_alerts.setdefault(monitor_id, {})
    wid = m.pop(key, None)
    return wid, ("end" if wid else None)


def build_interval_point(
    monitor_id: str,
    t: float,
    fhr_val: float,
    uterus_val: float,
    fhr_state_raw: int,
    fhr_baseline_event: float,
    toco_intensity: float,
    toco_alert_code: int,
    prev_fhr_line_status: int,
    prev_toco_code: int,
) -> Tuple[IntervalPoint, List[Dict], int, int]:
    """Строит точку и список предупреждений (в точке выставляется первое begin/end при переходах)."""
    # Разметка ЧСС (линия)
    fhr_line = FHR_STATE_MAP.get(int(fhr_state_raw), 0)
    # Разметка ЧСС (события)
    delta = float(fhr_val - fhr_baseline_event)
    fhr_evt = fhr_event_status(delta)
    # Разметка токограммы (линия)
    toco_line = toco_line_band(float(toco_intensity))
    # Фоны токограммы
    tachy = 1 if toco_alert_code == 1 else 0
    hyper = 1 if toco_alert_code == 2 else 0
    teta = 1 if toco_alert_code == 3 else 0

    warnings_here: List[Dict] = []
    warn_text = warn_id = warn_status = None

    # FHR переходы
    if fhr_line != prev_fhr_line_status:
        if prev_fhr_line_status != 0:
            key = f"fhr:{prev_fhr_line_status}"
            wid, st = end_alert(monitor_id, key)
            if wid:
                txt = f"{FHR_STATE_LABELS[prev_fhr_line_status]} — конец"
                warnings_here.append({"time_sec": t, "warning_id": wid, "status": st, "text": txt})
                warn_text = warn_text or txt; warn_id = warn_id or wid; warn_status = warn_status or st
        if fhr_line != 0:
            key = f"fhr:{fhr_line}"
            wid, st = begin_alert(monitor_id, key)
            txt = f"{FHR_STATE_LABELS[fhr_line]} — начало"
            warnings_here.append({"time_sec": t, "warning_id": wid, "status": st, "text": txt})
            warn_text = warn_text or txt; warn_id = warn_id or wid; warn_status = warn_status or st

    # TOCO переходы (по коду окна тревоги)
    if toco_alert_code != prev_toco_code:
        if prev_toco_code in ALERT_LABELS:
            key = f"toco:{prev_toco_code}"
            wid, st = end_alert(monitor_id, key)
            if wid:
                txt = f"{ALERT_LABELS[prev_toco_code]} — конец"
                warnings_here.append({"time_sec": t, "warning_id": wid, "status": st, "text": txt})
                warn_text = warn_text or txt; warn_id = warn_id or wid; warn_status = warn_status or st
        if toco_alert_code in ALERT_LABELS:
            key = f"toco:{toco_alert_code}"
            wid, st = begin_alert(monitor_id, key)
            txt = f"{ALERT_LABELS[toco_alert_code]} — начало"
            warnings_here.append({"time_sec": t, "warning_id": wid, "status": st, "text": txt})
            warn_text = warn_text or txt; warn_id = warn_id or wid; warn_status = warn_status or st

    point = IntervalPoint(
        monitor_id=monitor_id,
        time_sec=float(t),
        fhr_bpm=float(fhr_val),
        uterus_mmhg=float(uterus_val),
        warning_text=warn_text,
        warning_id=warn_id,
        warning_status=warn_status,
        fhr_line_status=int(fhr_line),
        fhr_event_status=int(fhr_evt),
        toco_line_status=int(toco_line),
        toco_tachysystole=int(tachy),
        toco_hypertonus=int(hyper),
        toco_tetanic=int(teta),
    )

    return point, warnings_here, fhr_line, toco_alert_code

# ————————————————————————————————————————————————————————————————
# Эндпоинты
# ————————————————————————————————————————————————————————————————

@app.get("/", tags=["Информация"])
async def root():
    return {
        "name": "КТГ Мониторинг API (upload-only)",
        "version": "2.0.0",
        "description": "Загружаете два CSV (ЧСС и матка) и получаете анализ.",
        "endpoints": {
            "streaming_upload": "POST /api/stream  (multipart/form-data: fhr_file, uterus_file)",
            "instant_upload": "POST /api/analyze (multipart/form-data: fhr_file, uterus_file)",
        },
        "csv_schema": {
            "required": ["value"],
            "optional": ["time_sec"],
            "notes": "Если 'time_sec' нет — берём равномерную шкалу с шагом 1/fs.",
        },
    }

@app.post("/api/stream", tags=["Потоковый анализ (SSE)"])
async def stream_analysis_upload(
    fhr_file: UploadFile = File(..., description="CSV с ЧСС плода (столбцы: value[, time_sec])"),
    uterus_file: UploadFile = File(..., description="CSV с маточной активностью (столбцы: value[, time_sec])"),
    interval_sec: float = Query(1.0, ge=0.1, le=10.0, description="Период отправки батчей (сек)"),
    fs: float = Query(FS_DEFAULT, gt=0.0, description="Частота дискретизации, Гц (если time_sec отсутствует)"),
):
    """
    Потоковая отправка данных КТГ батчами (Server-Sent Events).
    Загружаете два CSV в этом же запросе — ответом идёт SSE-поток.
    """
    monitor_id = str(uuid.uuid4())
    active_alerts[monitor_id] = {}

    # читаем и анализируем
    bpm_df = _read_csv_to_df(fhr_file, fs)
    uterus_df = _read_csv_to_df(uterus_file, fs)
    analysis = analyze_data(bpm_df, uterus_df, fs=fs)

    async def event_generator():
        total_points = len(analysis["time"])
        points_per_chunk = max(1, int(round(interval_sec * analysis["fs"])))

        prev_fhr_line = 0
        prev_toco_code = 0

        # служебное стартовое событие
        yield f'data: {{"monitor_id": "{monitor_id}", "status": "started", "total_points": {total_points}}}\n\n'

        for i in range(0, total_points, points_per_chunk):
            end_idx = min(i + points_per_chunk, total_points)
            batch_points: List[IntervalPoint] = []
            window_warnings: List[Dict] = []

            for idx in range(i, end_idx):
                pt, warns, prev_fhr_line, prev_toco_code = build_interval_point(
                    monitor_id=monitor_id,
                    t=float(analysis["time"][idx]),
                    fhr_val=float(analysis["fhr"]["data"][idx]),
                    uterus_val=float(analysis["toco"]["data"][idx]),
                    fhr_state_raw=int(analysis["fhr"]["states"][idx]),
                    fhr_baseline_event=float(analysis["fhr"]["baseline"][idx]),
                    toco_intensity=float(analysis["toco"]["intensity"][idx]),
                    toco_alert_code=int(analysis["toco"]["alert_code"][idx]),
                    prev_fhr_line_status=prev_fhr_line,
                    prev_toco_code=prev_toco_code,
                )
                batch_points.append(pt)
                if warns:
                    window_warnings.extend(warns)

            batch = StreamBatch(
                monitor_id=monitor_id,
                window_start_sec=float(analysis["time"][i]),
                window_end_sec=float(analysis["time"][end_idx - 1]),
                points=batch_points,
            )
            yield f"data: {batch.model_dump_json()}\n\n"
            await asyncio.sleep(interval_sec)

        # закрываем открытые предупреждения (end)
        for key, wid in list(active_alerts.get(monitor_id, {}).items()):
            txt = "Состояние — конец"
            yield f'data: {{"monitor_id":"{monitor_id}","warning_id":"{wid}","status":"end","text":"{txt}"}}\n\n'
        active_alerts.pop(monitor_id, None)

        yield f'data: {{"monitor_id": "{monitor_id}", "status": "completed"}}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",  # для SSE
        },
    )


@app.post("/api/analyze",
          response_model=InstantAnalysisResult,
          tags=["Мгновенный анализ"])
async def instant_analysis_upload(
    fhr_file: UploadFile = File(..., description="CSV с ЧСС плода (столбцы: value[, time_sec])"),
    uterus_file: UploadFile = File(..., description="CSV с маточной активностью (столбцы: value[, time_sec])"),
    fs: float = Query(FS_DEFAULT, gt=0.0, description="Частота дискретизации, Гц (если time_sec отсутствует)"),
):
    """
    Мгновенный анализ всего временного ряда КТГ по двум загруженным CSV.
    Возвращает полностью размеченный временной ряд и предупреждения begin/end.
    """
    monitor_id = str(uuid.uuid4())
    active_alerts[monitor_id] = {}

    bpm_df = _read_csv_to_df(fhr_file, fs)
    uterus_df = _read_csv_to_df(uterus_file, fs)
    analysis = analyze_data(bpm_df, uterus_df, fs=fs)

    timeseries: List[IntervalPoint] = []
    all_warnings: List[Dict] = []

    prev_fhr_line = 0
    prev_toco_code = 0

    for idx in range(len(analysis["time"])):
        pt, warns, prev_fhr_line, prev_toco_code = build_interval_point(
            monitor_id=monitor_id,
            t=float(analysis["time"][idx]),
            fhr_val=float(analysis["fhr"]["data"][idx]),
            uterus_val=float(analysis["toco"]["data"][idx]),
            fhr_state_raw=int(analysis["fhr"]["states"][idx]),
            fhr_baseline_event=float(analysis["fhr"]["baseline"][idx]),
            toco_intensity=float(analysis["toco"]["intensity"][idx]),
            toco_alert_code=int(analysis["toco"]["alert_code"][idx]),
            prev_fhr_line_status=prev_fhr_line,
            prev_toco_code=prev_toco_code,
        )
        timeseries.append(pt)
        if warns:
            all_warnings.extend(warns)

    # Закроем висящие предупреждения
    for key, wid in list(active_alerts.get(monitor_id, {}).items()):
        all_warnings.append({"time_sec": float(analysis["time"][-1]), "warning_id": wid, "status": "end", "text": "Состояние — конец"})
    active_alerts.pop(monitor_id, None)

    return InstantAnalysisResult(
        monitor_id=monitor_id,
        duration_sec=float(analysis["time"][-1]),
        timeseries=timeseries,
        warnings=all_warnings,
    )


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
