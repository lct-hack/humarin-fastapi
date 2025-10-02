import os
import math
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import json

from processing import (
    compute_signals_and_statuses,
    generate_warnings,
)

FS_DEFAULT = 7.87
ANNOTATION_REAL_PERIOD_SEC = 30.0 
ANNOTATION_MODEL_PERIOD_SEC = 30.0 
ANNOTATION_POLL_REAL_SEC = 0.3     


class Moment(BaseModel):
    monitor_id: str
    time_s: float
    real_time: str  # "HH:MM"
    fhr_bpm: Optional[float] = None
    uterus_data: Optional[float] = None
    stop: int = 0

class Batch(BaseModel):
    kind: Literal["moments_batch"] = "moments_batch"
    monitor_id: str
    t_start: float
    t_end: float
    moments: List[Moment]
    warnings: List[str] = Field(default_factory=list)

class StatusRange(BaseModel):
    start: float
    end: float
    color_id: int

class Annotation(BaseModel):
    kind: Literal["annotation"] = "annotation"
    monitor_id: str
    t_start: float
    t_end: float
    fhr_line_status: List[StatusRange]
    fhr_event_status: List[StatusRange]
    toco_line_status: List[StatusRange]
    toco_tachysystole: List[StatusRange]
    toco_hypertonus: List[StatusRange]
    toco_tetanic: List[StatusRange]
    warnings: List[str] = Field(default_factory=list) 

class MonitorSession:
    def __init__(self, monitor_id: str, fs: float, interval_sec: float, speed: float):
        self.monitor_id = monitor_id
        self.fs = fs
        self.interval_sec = interval_sec
        self.speed = speed
        self.created_at = datetime.now()

        self.time: Optional[np.ndarray] = None
        self.fhr: Optional[np.ndarray] = None
        self.uterus: Optional[np.ndarray] = None

        self.df_fhr: Optional[pd.DataFrame] = None  
        self.df_uterus: Optional[pd.DataFrame] = None 

        self.warnings_sorted: List[str] = []

        self.next_idx: int = 0
        self.points_per_batch: int = 1

        self.done: bool = False
        self.done_event: asyncio.Event = asyncio.Event()

        self.subscribers: List[asyncio.Queue] = []

        self.history: List[Moment] = []

        self.ann_last_t_end: float = 0.0
        self.ann_next_boundary: float = ANNOTATION_MODEL_PERIOD_SEC
        self.annotation_task: Optional[asyncio.Task] = None

    def has_data(self) -> bool:
        return (
            self.time is not None
            and self.fhr is not None
            and self.uterus is not None
            and self.df_fhr is not None
            and self.df_uterus is not None
        )

SESSIONS: Dict[str, MonitorSession] = {}

app = FastAPI(
    title="КТГ Мониторинг API (stream + annotations)",
    version="6.1.0",
    description=(
        "Эмуляция реального времени: RAW моменты идут сразу; "
        "аннотации приходят строго каждые 30 сек МОДЕЛЬНОГО времени (0..30, 0..60, …), "
        "в новом формате на интервалы: {start, end, color_id}."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _read_csv_to_df(upload: UploadFile, fs: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(upload.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV '{upload.filename}': {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail=f"Пустой файл: {upload.filename}")

    if "value" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "value"})
        else:
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' должен содержать столбец 'value' или один столбец.",
            )

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().all():
        raise HTTPException(status_code=400, detail=f"В '{upload.filename}' нет валидных числовых значений в 'value'.")

    if "time_sec" in df.columns:
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
        if df["time_sec"].isna().any():
            n = len(df)
            df["time_sec"] = np.arange(n) / float(fs)
    else:
        n = len(df)
        df["time_sec"] = np.arange(n) / float(fs)

    df = df.dropna(subset=["time_sec", "value"]).sort_values("time_sec").reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"После очистки в '{upload.filename}' не осталось данных.")
    return df[["time_sec", "value"]]

def _fmt_hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def _now() -> datetime:
    return datetime.now()

def _model_time_now(session: MonitorSession) -> float:
    elapsed_real = (_now() - session.created_at).total_seconds()
    return max(0.0, elapsed_real * max(0.1, session.speed))

def _subset_df_upto(df: pd.DataFrame, t_end: float) -> pd.DataFrame:
    """Отрезаем данные по времени <= t_end (включительно)."""
    return df.loc[df["time_sec"] <= (float(t_end) + 1e-9)].copy()

def _compress_status_ranges(t_arr: np.ndarray, status_arr: np.ndarray) -> List[StatusRange]:
    """Преобразует по-точечные статусы в интервалы [start,end] с color_id."""
    if t_arr is None or status_arr is None or len(t_arr) == 0:
        return []
    t = np.asarray(t_arr).astype(float)
    s = np.asarray(status_arr).astype(int)
    res: List[StatusRange] = []
    cur_val = int(s[0])
    seg_start = float(t[0])
    for i in range(1, len(s)):
        v = int(s[i])
        if v != cur_val:
            seg_end = float(t[i-1])
            res.append(StatusRange(start=seg_start, end=seg_end, color_id=cur_val))
            seg_start = float(t[i])
            cur_val = v
    res.append(StatusRange(start=seg_start, end=float(t[-1]), color_id=cur_val))
    return res

async def _publish_moments(session: MonitorSession, batch: Batch):
    session.history.extend(batch.moments)
    for q in list(session.subscribers):
        try:
            await q.put(batch.model_dump())
        except Exception:
            try:
                session.subscribers.remove(q)
            except ValueError:
                pass

async def _publish_annotation(session: MonitorSession, ann: Annotation):
    for q in list(session.subscribers):
        try:
            await q.put(ann.model_dump())
        except Exception:
            try:
                session.subscribers.remove(q)
            except ValueError:
                pass

async def _processing_loop(session: MonitorSession):
    """Часто шлём сырые моменты (без статусов)."""
    assert session.has_data()
    fs = session.fs
    dt_batch = session.interval_sec
    sleep_real = max(0.0, dt_batch / max(0.1, session.speed))
    session.points_per_batch = max(1, int(round(dt_batch * fs)))
    n = session.time.shape[0]

    while session.next_idx < n:
        i0 = session.next_idx
        i1 = min(n, i0 + session.points_per_batch)

        moments: List[Moment] = []
        for i in range(i0, i1):
            t_s = float(session.time[i])
            real_dt = session.created_at + timedelta(seconds=t_s / max(0.1, session.speed))
            moments.append(Moment(
                monitor_id=session.monitor_id,
                time_s=t_s,
                real_time=_fmt_hhmm(real_dt),
                fhr_bpm=float(session.fhr[i]) if session.fhr is not None else None,
                uterus_data=float(session.uterus[i]) if session.uterus is not None else None,
                stop=0,
            ))

        await _publish_moments(session, Batch(
            monitor_id=session.monitor_id,
            t_start=float(session.time[i0]),   
            t_end=float(session.time[i1 - 1]), 
            moments=moments,
            warnings=[],
        ))

        session.next_idx = i1
        if session.next_idx >= n:
            final_m = Moment(
                monitor_id=session.monitor_id,
                time_s=float(session.time[-1]),  
                real_time=_fmt_hhmm(
                    session.created_at + timedelta(seconds=float(session.time[-1]) / max(0.1, session.speed)) 
                ),
                fhr_bpm=float(session.fhr[-1]) if session.fhr is not None else None,
                uterus_data=float(session.uterus[-1]) if session.uterus is not None else None,
                stop=1,
            )
            await _publish_moments(session, Batch(
                monitor_id=session.monitor_id,
                t_start=float(session.time[-1]),   
                t_end=float(session.time[-1]),     
                moments=[final_m],
                warnings=[],
            ))
            session.done = True
            session.done_event.set()
            break

        if sleep_real > 0:
            await asyncio.sleep(sleep_real)

async def _annotation_loop(session: MonitorSession):
    """
    Строго каждые 30 с ПО РЯДУ шлём накопительную аннотацию 0..t_end.
    На каждой границе пересчитываем статусы ТОЛЬКО по прошедшим данным.
    При завершении потока досылаем финальную 0..last_time.
    Формат аннотаций — интервалы {start, end, color_id}.
    """
    assert session.has_data()
    full_last_time = float(session.time[-1])  
    window = float(ANNOTATION_MODEL_PERIOD_SEC)

    while True:
        cur_model = min(_model_time_now(session), full_last_time)

        boundary = session.ann_next_boundary

        need_flush_final = session.done_event.is_set() and (session.ann_last_t_end < full_last_time - 1e-9)
        reached_boundary = cur_model >= boundary - 1e-9

        if reached_boundary or need_flush_final:
            t_end = full_last_time if need_flush_final and not reached_boundary else min(boundary, full_last_time)

            df_fhr_sub = _subset_df_upto(session.df_fhr, t_end)
            df_uter_sub = _subset_df_upto(session.df_uterus, t_end)

            if min(len(df_fhr_sub), len(df_uter_sub)) >= 2:
                t_arr, _fhr, _uter, st, warns_sorted = compute_signals_and_statuses(
                    df_fhr_sub, df_uter_sub, fs=session.fs
                )

                ann = Annotation(
                    monitor_id=session.monitor_id,
                    t_start=float(t_arr[0]),
                    t_end=float(t_arr[-1]),
                    fhr_line_status=_compress_status_ranges(t_arr, st["fhr_line_status"]),
                    fhr_event_status=_compress_status_ranges(t_arr, st["fhr_event_status"]),
                    toco_line_status=_compress_status_ranges(t_arr, st["toco_line_status"]),
                    toco_tachysystole=_compress_status_ranges(t_arr, st["toco_tachysystole"]),
                    toco_hypertonus=_compress_status_ranges(t_arr, st["toco_hypertonus"]),
                    toco_tetanic=_compress_status_ranges(t_arr, st["toco_tetanic"]),
                    warnings=list(warns_sorted) if warns_sorted else [],
                )
                await _publish_annotation(session, ann)

                session.ann_last_t_end = float(t_arr[-1])

            if reached_boundary and session.ann_next_boundary < full_last_time - 1e-9:
                session.ann_next_boundary += window

            if session.done_event.is_set() and session.ann_last_t_end >= full_last_time - 1e-9:
                break

        await asyncio.sleep(ANNOTATION_POLL_REAL_SEC)

@app.get("/")
def root():
    return {
        "name": "КТГ Мониторинг API",
        "version": "0.5",
        "endpoints": {
            "upload": "POST /api/upload",
            "stream": "GET  /api/stream/{monitor_id}",
            "monitors": "GET  /api/monitors",
            "instant": "POST /api/instant",
        },
        "notes": {
            "annotation_period_model_sec": ANNOTATION_MODEL_PERIOD_SEC,
            "annotation_period_real_sec_legacy": ANNOTATION_REAL_PERIOD_SEC,
            "annotation_mode": "cumulative_0_to_t_end",
            "stream_message_kinds": ["moments_batch", "annotation"],
            "annotation_format": "status_as_ranges",
        },
    }

@app.get("/api/monitors")
def list_monitors():
    return {
        "active": [mid for (mid, s) in SESSIONS.items() if not s.done],
        "finished": [mid for (mid, s) in SESSIONS.items() if s.done],
    }

@app.post("/api/upload")
async def upload_and_start(
    fhr_file: UploadFile = File(..., description="CSV с ЧСС (value[, time_sec])"),
    uterus_file: UploadFile = File(..., description="CSV с маткой (value[, time_sec])"),
    monitor_id: Optional[str] = Query(None, description="Явный ID монитора (если не указан — будет сгенерирован)"),
    interval_sec: float = Query(1.0, ge=0.1, le=10.0, description="Период батча сырья, сек"),
    fs: float = Query(FS_DEFAULT, gt=0.0, description="Частота дискретизации (если нет time_sec)"),
    speed: float = Query(1.0, ge=0.1, le=100.0, description="Ускорение модельного времени"),
):
    """
    Запускаем два фоновых обработчика:
      1) поток сырых моментов;
      2) поток аннотаций (каждые 30с ПО РЯДУ, накопительно 0..t_end, с пересчётом).
    Формат аннотаций: интервалы {start,end,color_id} по каждому статусу.
    """
    mid = monitor_id or str(uuid.uuid4())
    if mid in SESSIONS and not SESSIONS[mid].done:
        raise HTTPException(status_code=409, detail=f"monitor_id '{mid}' уже активен")

    bpm_df = _read_csv_to_df(fhr_file, fs)
    uter_df = _read_csv_to_df(uterus_file, fs)
    if min(len(bpm_df), len(uter_df)) < 10:
        raise HTTPException(status_code=400, detail="Слишком мало данных")

    time_arr, fhr, uterus, _statuses, warnings_sorted = compute_signals_and_statuses(
        bpm_df, uter_df, fs=fs
    )

    session = MonitorSession(mid, fs, interval_sec, speed)
    session.time = time_arr
    session.fhr = fhr
    session.uterus = uterus

    session.df_fhr = bpm_df
    session.df_uterus = uter_df

    session.warnings_sorted = list(warnings_sorted) if warnings_sorted else []

    SESSIONS[mid] = session
    asyncio.create_task(_processing_loop(session))
    session.annotation_task = asyncio.create_task(_annotation_loop(session))

    return {
        "monitor_id": mid,
        "points": int(time_arr.shape[0]),
        "interval_sec": interval_sec,
        "speed": speed,
        "annotation_period_model_sec": ANNOTATION_MODEL_PERIOD_SEC,
        "annotation_mode": "cumulative_0_to_t_end",
        "annotation_format": "status_as_ranges",
        "stream_url": f"/api/stream/{mid}",
    }

@app.get("/api/stream/{monitor_id}")
async def connect_stream(monitor_id: str = Path(..., description="ID монитора, ранее выданный /api/upload")):
    if monitor_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="monitor_id не найден")

    session = SESSIONS[monitor_id]
    q: asyncio.Queue = asyncio.Queue()
    session.subscribers.append(q)

    async def event_gen():
        if session.history:
            hist_batch = Batch(
                monitor_id=session.monitor_id,
                t_start=float(session.history[0].time_s),
                t_end=float(session.history[-1].time_s),
                moments=session.history,
                warnings=[],
            )
            yield f"data: {hist_batch.model_dump_json()}\n\n"

        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                session.subscribers.remove(q)
            except ValueError:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream; charset=utf-8", headers=headers)

@app.post("/api/instant")
async def instant(
    fhr_file: UploadFile = File(..., description="CSV с ЧСС (value[, time_sec])"),
    uterus_file: UploadFile = File(..., description="CSV с маткой (value[, time_sec])"),
    monitor_id: Optional[str] = Query(None),
    fs: float = Query(FS_DEFAULT, gt=0.0),
    interval_sec: float = Query(1.0, ge=0.1, le=10.0, description="Для формирования real_time в 'moments'"),
    speed: float = Query(1.0, ge=0.1, le=100.0),
):
    """
    Мгновенная обработка:
      - 'moments' — полный список точек без статусов;
      - 'annotations' — одна полная аннотация 0..total в формате интервалов.
    """
    mid = monitor_id or str(uuid.uuid4())

    bpm_df = _read_csv_to_df(fhr_file, fs)
    uter_df = _read_csv_to_df(uterus_file, fs)

    time_arr_full, fhr_full, uter_full, _st_full, _ = compute_signals_and_statuses(
        bpm_df, uter_df, fs=fs
    )

    created_at = _now()

    moments: List[Dict[str, Any]] = []
    for i in range(time_arr_full.shape[0]):
        t_s = float(time_arr_full[i])
        real_dt = created_at + timedelta(seconds=t_s / max(0.1, speed))
        moments.append({
            "monitor_id": mid,
            "time_s": t_s,
            "real_time": _fmt_hhmm(real_dt),
            "fhr_bpm": float(fhr_full[i]),
            "uterus_data": float(uter_full[i]),
            "stop": 0,
        })
    if moments:
        moments[-1]["stop"] = 1

    # одна полная аннотация 0..total
    total = float(time_arr_full[-1])
    annotations: List[Dict[str, Any]] = []
    df_fhr_sub = _subset_df_upto(bpm_df, total)
    df_uter_sub = _subset_df_upto(uter_df, total)
    if min(len(df_fhr_sub), len(df_uter_sub)) >= 2:
        t_arr, _fhr, _uter, st, warns_sorted = compute_signals_and_statuses(
            df_fhr_sub, df_uter_sub, fs=fs
        )
        ann = Annotation(
            monitor_id=mid,
            t_start=float(t_arr[0]),
            t_end=float(t_arr[-1]),
            fhr_line_status=_compress_status_ranges(t_arr, st["fhr_line_status"]),
            fhr_event_status=_compress_status_ranges(t_arr, st["fhr_event_status"]),
            toco_line_status=_compress_status_ranges(t_arr, st["toco_line_status"]),
            toco_tachysystole=_compress_status_ranges(t_arr, st["toco_tachysystole"]),
            toco_hypertonus=_compress_status_ranges(t_arr, st["toco_hypertonus"]),
            toco_tetanic=_compress_status_ranges(t_arr, st["toco_tetanic"]),
            warnings=list(warns_sorted) if warns_sorted else [],
        )
        annotations = [ann.model_dump()]

    return {
        "monitor_id": mid,
        "duration_sec": total,
        "moments": moments,
        "annotations": annotations,
        "annotation_period_model_sec": ANNOTATION_MODEL_PERIOD_SEC,
        "annotation_mode": "full_0_to_total",
        "annotation_format": "status_as_ranges",
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
