from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

def _interp_nans(x: np.ndarray) -> np.ndarray:
    y = x.astype(float).copy()
    n = y.size
    if n == 0:
        return y
    mask = np.isnan(y)
    if not mask.any():
        return y
    idx = np.arange(n, dtype=float)
    y[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return y

def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if w <= 1:
        return x.copy()
    k = np.ones(w, float)
    s = np.convolve(x, k, "same")
    c = np.convolve(np.ones_like(x), k, "same")
    return s / np.maximum(c, 1e-9)


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel() 
    w = max(1, int(w))
    if w == 1 or x.size < w:
        return x.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(x, w)
    med = np.median(sw, axis=-1)
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    return np.pad(med, (pad_l, pad_r), mode="edge")

def _hampel(x: np.ndarray, w: int, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Классический Hampel: заменяет выбросы на локальную медиану.
    w — нечётное окно в пробах.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = max(3, int(w) | 1) 
    from numpy.lib.stride_tricks import sliding_window_view

    sw = sliding_window_view(x, w) 
    med = np.median(sw, axis=-1)
    mad = np.median(np.abs(sw - med[:, None]), axis=-1)  

    pad_l = w // 2
    pad_r = w - 1 - pad_l
    med_f = np.pad(med, (pad_l, pad_r), mode="edge")
    mad_f = np.pad(mad, (pad_l, pad_r), mode="edge")

    thresh = n_sigmas * 1.4826 * np.maximum(mad_f, 1e-9)

    y = x.copy()
    outliers = np.abs(y - med_f) > thresh
    y[outliers] = med_f[outliers]
    return y

def _fhr_event_status(delta: float, enter: float = 12.0) -> int:
    if delta >= enter: return 1
    if delta <= -enter: return -1
    return 0

def _toco_line_band(intensity: float) -> int:
    if intensity > 80.0: return 2
    if intensity >= 30.0: return 1
    return 0

def _classify_fhr_states(
    fhr: np.ndarray,
    fs: float,
    *,
    smooth_seconds: float = 5.0,
    hampel_seconds: float = 3.0,
    sustain_seconds: float = 600.0,
    event_baseline_seconds: float = 600.0,
    severe_brady_threshold: float = 100.0,
    brady_threshold: float = 120.0,
    tachy_threshold: float = 160.0,
    severe_tachy_threshold: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает:
      states         — коды 1..5 (как в processing.py)
      baseline_event — локальная базальная линия для событий (скользящая медиана)
      x_smooth       — сглаженный и «очищенный» ряд (для последующей детекции событий)
    """
    x = _interp_nans(np.asarray(fhr, dtype=float).ravel())

    # сглаживание + Hampel (как в processing.py)
    smooth_win = max(1, int(round(fs * smooth_seconds)))
    x_smooth = _moving_avg(x, smooth_win)
    x_smooth = _hampel(x_smooth, max(3, int(round(fs * hampel_seconds)) | 1))

    # «устойчивая» база для КЛАССИФИКАЦИИ состояний
    sustain_win = max(1, int(round(fs * sustain_seconds)))
    baseline_state = _moving_avg(x_smooth, sustain_win)

    # состояния 1..5 (точно как во втором файле)
    states = np.full_like(x, 0, dtype=int)

    mask1 = (baseline_state < brady_threshold)
    mask2 = (baseline_state < severe_brady_threshold)
    mask3 = (baseline_state > tachy_threshold)
    mask4 = (baseline_state > severe_tachy_threshold)

    min_len = min(len(states), len(mask1))
    states = states[:min_len]

    if len(mask1) != min_len: mask1 = mask1[:min_len]
    if len(mask2) != min_len: mask2 = mask2[:min_len]
    if len(mask3) != min_len: mask3 = mask3[:min_len]
    if len(mask4) != min_len: mask4 = mask4[:min_len]

    states[mask1] = -1
    states[mask2] = -2
    states[mask3] = 1
    states[mask4] = 2

    ev_win = max(1, int(round(fs * event_baseline_seconds)))
    baseline_event = _rolling_median(x_smooth, ev_win)

    n = x.size
    if baseline_event.size != n:
        baseline_event = baseline_event[:n]
    if states.size != n:
        states = states[:n]
    if x_smooth.size != n:
        x_smooth = x_smooth[:n]

    return states, baseline_event, x_smooth
def _align_len(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    m = min(len(a) for a in arrs)
    return tuple(a[:m] for a in arrs)

def _segments_from_mask(mask: np.ndarray) -> List[tuple]:
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts, ends))

def _merge_segments(segs: List[tuple], max_gap: int) -> List[tuple]:
    if not segs:
        return []
    segs.sort()
    res = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = res[-1]
        if s - pe <= max_gap:
            res[-1] = (ps, max(pe, e))
        else:
            res.append((s, e))
    return res

def _filter_by_len_and_peak(delta: np.ndarray, segs: List[tuple],
                            min_len: int, min_peak: float, kind: str) -> List[tuple]:
    out = []
    for s, e in segs:
        if e - s < min_len:
            continue
        seg = delta[s:e]
        peak = np.max(seg) if kind == "accel" else -np.min(seg)
        if peak >= min_peak:
            out.append((s, e))
    return out

def _hysteresis_mask(delta: np.ndarray, enter_thr: float, exit_thr: float, sign: int) -> np.ndarray:
    assert sign in (+1, -1)
    x = sign * delta
    m = np.zeros_like(x, dtype=bool)
    on = False
    for i, v in enumerate(x):
        if not on and v >= enter_thr:
            on = True
        if on:
            m[i] = True
            if v < exit_thr:
                on = False
    return m


def _analyze_toco(uterus: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляем:
      intensity (excess >= 0)
      тахисистолия/гипертонус/тетания (простая эвристика)
    """
    x = _interp_nans(uterus)
    xs = _moving_avg(x, max(1, int(round(fs * 3.0))))
    tone = _rolling_median(xs, max(1, int(round(fs * 75.0))))
    excess = np.maximum(xs - tone, 0.0)

    on = excess >= 10.0
    segs = []
    cur = None
    for i, v in enumerate(on):
        if v and cur is None:
            cur = i
        if (not v or i == on.size - 1) and cur is not None:
            j = i if not v else i + 1
            if (j - cur) / fs >= 120.0:
                segs.append((cur, j))
            cur = None

    n = x.size
    alert_tet = np.zeros(n, dtype=int)
    for s, e in segs:
        alert_tet[s:e] = 1

    L = int(round(600.0 * fs))
    S = int(round(60.0 * fs))
    starts = list(range(0, max(1, n - L + 1), S)) or [0]

    peaks = (excess[1:-1] > excess[:-2]) & (excess[1:-1] >= excess[2:]) & (excess[1:-1] >= 10.0)
    peak_idx = np.flatnonzero(peaks) + 1

    alert_tachy = np.zeros(n, dtype=int)
    alert_hyper = np.zeros(n, dtype=int)
    for s0 in starts:
        e0 = min(n, s0 + L)

        cnt = int(np.sum((peak_idx >= s0) & (peak_idx < e0)))
        if cnt > 5:
            alert_tachy[s0:e0] = 1

        if e0 > s0 and float(np.median(tone[s0:e0])) > 25.0:
            alert_hyper[s0:e0] = 1

    intensity = excess * 1.0
    return intensity, alert_tachy, alert_hyper | alert_tet 

STATE_LABELS: Dict[int, str] = {
    0: "Норма",
    -1: "Умеренная брадикардия",
    -2: "Тяжелая брадикардия",
    1: "Умеренная тахикардия",
    2: "Тяжелая тахикардия",
}

def compute_signals_and_statuses(
    bpm_df: pd.DataFrame,
    uterus_df: pd.DataFrame,
    fs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Выравнивает длины входов/промежуточных рядов, считает статусы на КАЖДОЙ точке.
    Возвращает:
      time_s, fhr, uterus,
      statuses: dict (линии статусов + события + коды состояний -2..2 + СТАТИСТИКА),
      warnings_sorted: List[str]
    """

    def _align_len(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
        if not arrs:
            return tuple()
        m = min(len(a) for a in arrs)
        return tuple(a[:m] for a in arrs)

    n0 = min(len(bpm_df), len(uterus_df))
    if n0 <= 0:
        empty_f = np.array([], dtype=float)
        empty_i = np.array([], dtype=int)
        statistics = {STATE_LABELS[k]: 0.0 for k in STATE_LABELS}
        statistics["Всего, сек"] = 0.0
        statuses = {
            "fhr_line_status": empty_i,
            "fhr_event_status": empty_i,
            "fhr_states": empty_i,
            "toco_line_status": empty_i,
            "toco_tachysystole": empty_i,
            "toco_hypertonus": empty_i,
            "toco_tetanic": empty_i,
            "fhr_statistics": statistics,
            "baseline_event": empty_f,
        }
        return empty_f, empty_f, empty_f, statuses, ["Нет данных"]

    bpm_df = bpm_df.iloc[:n0].reset_index(drop=True)
    uterus_df = uterus_df.iloc[:n0].reset_index(drop=True)

    time_s = bpm_df["time_sec"].values.astype(float, copy=False)
    fhr = bpm_df["value"].values.astype(float, copy=False)
    uterus = uterus_df["value"].values.astype(float, copy=False)

    states, baseline_ev, x_smooth = _classify_fhr_states(fhr, fs)

    time_s, fhr, uterus = _align_len(time_s, fhr, uterus)
    states, baseline_ev, x_smooth = _align_len(states, baseline_ev, x_smooth)
    n = min(len(time_s), len(fhr), len(uterus), len(states), len(baseline_ev), len(x_smooth))
    time_s, fhr, uterus, states, baseline_ev, x_smooth = (
        time_s[:n], fhr[:n], uterus[:n], states[:n], baseline_ev[:n], x_smooth[:n]
    )

    fhr_line = states.astype(int, copy=False)

    delta = x_smooth - baseline_ev
    accel_mask0 = _hysteresis_mask(delta, enter_thr=12.0, exit_thr=8.0, sign=+1)
    decel_mask0 = _hysteresis_mask(delta, enter_thr=12.0, exit_thr=8.0, sign=-1)

    merge_gap = int(round(fs * 5.0))
    accel_segs = _merge_segments(_segments_from_mask(accel_mask0), merge_gap)
    decel_segs = _merge_segments(_segments_from_mask(decel_mask0), merge_gap)

    min_len_acc = int(round(fs * 15.0))
    min_len_dec = int(round(fs * 15.0))
    accel_segs = _filter_by_len_and_peak(delta, accel_segs, min_len_acc, 15.0, "accel")
    decel_segs = _filter_by_len_and_peak(delta, decel_segs, min_len_dec, 15.0, "decel")

    fhr_evt = np.zeros(n, dtype=int)
    for s, e in accel_segs:
        s0, e0 = max(0, int(s)), min(n, int(e))
        if e0 > s0:
            fhr_evt[s0:e0] = 1
    for s, e in decel_segs:
        s0, e0 = max(0, int(s)), min(n, int(e))
        if e0 > s0:
            fhr_evt[s0:e0] = -1

    intensity, alert_tachy, alert_hyper_or_tet = _analyze_toco(uterus, fs)
    (intensity, alert_tachy, alert_hyper_or_tet) = _align_len(intensity, alert_tachy, alert_hyper_or_tet)
    intensity = intensity[:n]; alert_tachy = alert_tachy[:n]; alert_hyper_or_tet = alert_hyper_or_tet[:n]

    toco_line = np.array([_toco_line_band(float(v)) for v in intensity], dtype=int)
    win = max(1, int(round(10 * fs)))
    mean_int = _moving_avg(intensity, win)
    if len(mean_int) != n:
        mean_int = mean_int[:n]

    toco_tet = np.zeros_like(toco_line, dtype=int)
    toco_tet[(alert_hyper_or_tet == 1) & (mean_int > 40.0)] = 1
    toco_hyper = np.clip(alert_hyper_or_tet - toco_tet, 0, 1)

    statistics: Dict[str, float] = {STATE_LABELS[s]: float(np.sum(states == s) / fs) for s in STATE_LABELS}
    statistics["Всего, сек"] = float(n / fs)

    if n > 0:
        cp = np.flatnonzero(np.diff(states)) + 1
        seg_starts = np.r_[0, cp]
        seg_states = states[seg_starts]
    else:
        seg_states = np.array([], dtype=int)

    for s in (-2, -1, 1, 2):
        statistics[f"Промежутки (шт) — {STATE_LABELS[s]}"] = float(int(np.sum(seg_states == s)))
    statistics["Промежутки (шт) — Не норма (всего)"] = float(int(np.sum(seg_states != 0)))

    def _dur(segs: List[tuple]) -> float:
        return sum((e - s) for s, e in segs) / fs

    def _amps(segs: List[tuple], kind: str) -> List[float]:
        vals: List[float] = []
        for s, e in segs:
            s0, e0 = max(0, int(s)), min(n, int(e))
            if e0 <= s0:
                continue
            seg = delta[s0:e0]
            vals.append(float(np.max(seg) if kind == "accel" else -np.min(seg)))
        return vals

    acc_durs = [(min(n, e) - max(0, s)) / fs for s, e in accel_segs if min(n, e) > max(0, s)]
    dec_durs = [(min(n, e) - max(0, s)) / fs for s, e in decel_segs if min(n, e) > max(0, s)]
    acc_amps = _amps(accel_segs, "accel")
    dec_deps = _amps(decel_segs, "decel")

    statistics.update({
        "Акселерации — всего, сек": float(_dur(accel_segs)),
        "Акселерации — эпизоды (шт)": float(len(accel_segs)),
        "Акселерации — медиана длительности, сек": float(np.median(acc_durs) if acc_durs else 0.0),
        "Акселерации — макс амплитуда, уд/мин": float(max(acc_amps) if acc_amps else 0.0),

        "Децелерации — всего, сек": float(_dur(decel_segs)),
        "Децелерации — эпизоды (шт)": float(len(decel_segs)),
        "Децелерации — медиана длительности, сек": float(np.median(dec_durs) if dec_durs else 0.0),
        "Децелерации — макс глубина, уд/мин": float(max(dec_deps) if dec_deps else 0.0),

        "Децелерации — пролонгированные >2 мин (шт)": float(sum(d > 120.0 for d in dec_durs)),
        "Децелерации — ≥5 мин (шт)": float(sum(d >= 300.0 for d in dec_durs)),
    })

    statuses: Dict[str, Any] = {
        "fhr_line_status": fhr_line.astype(int, copy=False),
        "fhr_event_status": fhr_evt.astype(int, copy=False),
        "fhr_states": states.astype(int, copy=False),           
        "toco_line_status": toco_line.astype(int, copy=False),   
        "toco_tachysystole": alert_tachy.astype(int, copy=False),
        "toco_hypertonus": toco_hyper.astype(int, copy=False),
        "toco_tetanic": toco_tet.astype(int, copy=False),
        "fhr_statistics": statistics,
        "baseline_event": baseline_ev.astype(float, copy=False),
    }

    warnings_sorted = generate_warnings(
        fhr=fhr, uterus=uterus, fs=fs,
        fhr_line=fhr_line, fhr_event=fhr_evt,
        toco_line=toco_line, toco_tachy=alert_tachy,
        toco_hyper=toco_hyper, toco_tet=toco_tet
    )

    return time_s, fhr, uterus, statuses, warnings_sorted



# ————————————————————————————————————————————————————————————————
# Генератор предупреждений (только здесь!)
# ————————————————————————————————————————————————————————————————

def generate_warnings(
    *,
    fhr: np.ndarray,
    uterus: np.ndarray,
    fs: float,
    fhr_line: np.ndarray,
    fhr_event: np.ndarray,
    toco_line: np.ndarray,
    toco_tachy: np.ndarray,
    toco_hyper: np.ndarray,
    toco_tet: np.ndarray,
) -> List[str]:
    """
    Совместимо по сигнатуре, но логика — как в rules 6.1–6.5 из processing.py.
    Итог — уникальные тексты WARNING/CRITICAL, отсортированные по важности.
    """
    import numpy as np

    # --- FHR препроцессинг (ровно как во втором файле) ---
    x = _interp_nans(np.asarray(fhr, float).ravel())
    smooth_win = max(1, int(round(fs * 5.0)))
    x_smooth = _moving_avg(x, smooth_win)
    x_smooth = _hampel(x_smooth, max(3, int(round(fs * 3.0)) | 1))

    ev_win = max(1, int(round(fs * 600.0)))
    baseline_event = _rolling_median(x_smooth, ev_win)
    delta = x_smooth - baseline_event

    # Децелерации: гистерезис → merge → фильтры (как в processing.py)
    dec_mask0 = _hysteresis_mask(delta, enter_thr=12.0, exit_thr=8.0, sign=-1)
    dec_segs = _merge_segments(_segments_from_mask(dec_mask0), int(round(fs * 5.0)))
    dec_segs = _filter_by_len_and_peak(delta, dec_segs, int(round(fs * 15.0)), 15.0, "decel")
    dec_segs_s = [(s / fs, e / fs) for s, e in dec_segs]
    dec_long_warn = [(s, e) for (s, e) in dec_segs_s if (e - s) > 90.0]
    dec_long_crit = [(s, e) for (s, e) in dec_segs_s if (e - s) >= 180.0]

    # Устойчивые сегменты тяжёлой брадикардии ≥3 мин (6.1)
    def _sustained(mask_func, min_s: float) -> List[tuple]:
        m = mask_func(x_smooth)
        out = []
        for s, e in _segments_from_mask(m):
            if (e - s) / fs >= min_s:
                out.append((s / fs, e / fs))
        return out

    sev_brady_segs = _sustained(lambda v: v < 100.0, 180.0)
    
    # --- ТОКО: восстановим схватки и окна так же, как во втором файле ---
    # Сгладим и отделим «тонус»
    xs = _moving_avg(_interp_nans(np.asarray(uterus, float).ravel()),
                     max(1, int(round(fs * 3.0))))
    tone = _rolling_median(xs, max(1, int(round(fs * 75.0))))
    excess = np.maximum(xs - tone, 0.0)

    # Схватки: excess>=10 мм рт.ст., ≥30 с, «хвост» завершения 12 с
    thr = 10.0
    min_len = int(round(fs * 30.0))
    gap_close = int(round(fs * 12.0))
    raw = _segments_from_mask(excess >= thr)
    merged = []
    if raw:
        cs, ce = raw[0]
        for s, e in raw[1:]:
            if s - ce <= gap_close: ce = e
            else: merged.append((cs, ce)); cs, ce = s, e
        merged.append((cs, ce))
    contr = [(s, e) for s, e in merged if (e - s) >= min_len]
    contr_s = np.array([s / fs for s, e in contr]) if contr else np.array([], float)
    contr_e = np.array([e / fs for s, e in contr]) if contr else np.array([], float)

    # Окна 10 мин, шаг 60 с → «тахисистолия» (>5 пиков) и «гипертонус» (median tone >25)
    L = int(round(600.0 * fs)); S = int(round(60.0 * fs)); n = len(xs)
    starts = list(range(0, max(1, n - L + 1), S)) or [0]
    # пики схваток (по excess)
    peaks = (excess[1:-1] > excess[:-2]) & (excess[1:-1] >= excess[2:]) & (excess[1:-1] >= 10.0)
    peak_idx = (np.flatnonzero(peaks) + 1).astype(int)

    windows = []
    for s0 in starts:
        e0 = min(n, s0 + L)
        cnt = int(np.sum((peak_idx >= s0) & (peak_idx < e0)))
        tone_med = float(np.median(tone[s0:e0])) if e0 > s0 else 0.0
        lab = None
        if cnt > 5: lab = "Тахисистолия"
        elif tone_med > 25.0: lab = "Гипертонус"
        windows.append((s0 / fs, e0 / fs, lab))

    # Тетания: схватки ≥120 с
    tetanic = [(s / fs, e / fs) for s, e in contr if (e - s) / fs >= 120.0]

    def _overlap(a0,a1,b0,b1): return (min(a1,b1) - max(a0,b0)) > 0.0

    # Вспомогательные функции для правил 6.4–6.5
    def ratio_decels_per_contr(t0: float, t1: float) -> float:
        if contr_s.size == 0: return 0.0
        idx = np.where((contr_s < t1) & (contr_e > t0))[0]
        if idx.size == 0: return 0.0
        coupled = 0
        for k in idx:
            c0, c1 = float(contr_s[k]), float(contr_e[k])
            if any(_overlap(c0, c1, d0, d1) for (d0, d1) in dec_segs_s):
                coupled += 1
        return coupled / float(idx.size)

    def tachysystole_recent(t: float, horizon_s: float = 1800.0) -> bool:
        a0, a1 = max(0.0, t - horizon_s), t
        return any((lab == "Тахисистолия") and _overlap(a0, a1, w0, w1)
                   for (w0, w1, lab) in windows)

    def hyper_overlap(t0: float, t1: float) -> bool:
        return any((lab == "Гипертонус") and _overlap(t0, t1, w0, w1)
                   for (w0, w1, lab) in windows)

    def no_recovery_between_contractions(t0: float, t1: float) -> bool:
        if contr_s.size < 2: return False
        idx = np.where((contr_s >= t0) & (contr_e <= t1))[0]
        if idx.size < 2: return False
        for i in range(idx.size - 1):
            e1 = contr_e[idx[i]]; s2 = contr_s[idx[i+1]]
            lo = int(max(0, np.floor(e1 * fs))); hi = int(min(len(x_smooth), np.ceil(s2 * fs)))
            if hi > lo and np.min(x_smooth[lo:hi]) < 120.0:
                return True
            for (d0, d1) in dec_segs_s:
                if _overlap(e1, s2, d0, d1): return True
        return False

    # --- Правила 6.1–6.5 → тексты ---
    texts: list[tuple[int, str]] = []

    if sev_brady_segs:
        print("LOLL")
        texts.append((0, "КРИТИЧНО: Тяжёлая брадикардия ≥3 мин"))

    if dec_long_crit:
        texts.append((0, "КРИТИЧНО: Децелерация ≥3 мин"))

    for (u0, u1) in tetanic:
        # падение ЧСС/децелер. или гипертонус в интервале тетании
        lo = int(max(0, np.floor(u0 * fs))); hi = int(min(len(x_smooth), np.ceil(u1 * fs)))
        fhr_drop = (hi > lo) and (np.min(x_smooth[lo:hi]) < 120.0)
        dec_overlap = any(_overlap(u0, u1, d0, d1) for (d0, d1) in dec_segs_s)
        if fhr_drop or dec_overlap or hyper_overlap(u0, u1):
            texts.append((1, "КРИТИЧНО: Тетания/гипертонус + падение ЧСС/децелерации"))
            break

    T_total = len(x_smooth) / fs
    win = 1200.0; step = 60.0
    t = 0.0; seen_64 = False; seen_65_crit = False; seen_65_warn = False
    while t < T_total:
        w0, w1 = t, min(t + win, T_total)
        ratio = ratio_decels_per_contr(w0, w1)
        if ratio >= 0.5 and not seen_64 and tachysystole_recent(w1, 1800.0):
            texts.append((1, "КРИТИЧНО: Тахисистолия (посл.30 мин) + децелерации в ≥50% схваток (20 мин)"))
            seen_64 = True
        if ratio >= 0.5:
            no_rec = no_recovery_between_contractions(w0, w1)
            long_present = any(_overlap(w0, w1, s, e) for (s, e) in dec_long_warn)
            if (no_rec or long_present) and not seen_65_crit:
                texts.append((2, "КРИТИЧНО: ≥50% децелераций + (нет восстановления или длительные >90 с)"))
                seen_65_crit = True
            elif not seen_65_warn and not (no_rec or long_present):
                texts.append((4, "ВНИМАНИЕ: Децелерации в ≥50% схваток (20 мин)"))
                seen_65_warn = True
        t += step

    # Если ничего критичного/предупреждающего не нашли:
    if not texts:
        return ["Норма"]

    texts.sort(key=lambda x: x[0])
    # уникальные формулировки по приоритету
    out = []
    for _, s in texts:
        if s not in out:
            out.append(s)
    return out


def _max_run_len(mask01: np.ndarray, fs: float) -> float:
    """Максимальная длина непрерывного участка (сек) в 0/1 маске."""
    max_len = 0
    cur = 0
    for v in mask01:
        if v:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return max_len / fs

def _share_on(mask01: np.ndarray) -> float:
    return float(np.mean(mask01.astype(bool)))

def _long_events_share(event_code: np.ndarray, fs: float, *, sign: int, min_s: float) -> bool:
    """Есть ли события указанного знака длительностью более min_s секунд."""
    assert sign in (-1, 1)
    m = (event_code == sign).astype(int)
    return _max_run_len(m, fs) >= min_s
