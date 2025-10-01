import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

STATE_LABELS: Dict[int, str] = {
    1: "Норма",
    2: "Умеренная брадикардия",
    3: "Тяжелая брадикардия",
    4: "Умеренная тахикардия",
    5: "Тяжелая тахикардия",
}
STATE_COLORS: Dict[int, str] = {
    1: "#2ecc71",
    2: "#3498db",
    3: "#0b3c5d",
    4: "#f39c12",
    5: "#e74c3c",
}
BG_ACCEL = "#F5C2C7"
BG_DECEL = "#AED6F1"
BG_DECEL_LONG = "#85C1E9"
def _resample_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    """Линейно приводит x к длине target_len (если уже совпадает — возвращает как есть)."""
    x = np.asarray(x, float).ravel()
    if x.size == target_len:
        return x
    if target_len <= 0 or x.size == 0:
        return np.zeros(max(0, target_len), dtype=float)
    src = np.arange(x.size, dtype=float)
    dst = np.linspace(0, x.size - 1, target_len, dtype=float)
    return np.interp(dst, src, x)
def _moving_avg(x: np.ndarray, win_samples: int) -> np.ndarray:
    if win_samples < 1:
        return x.copy()
    k = np.ones(win_samples, float)
    s = np.convolve(x, k, "same")
    c = np.convolve(np.ones_like(x), k, "same")
    return s / np.maximum(c, 1e-9)

def _rolling_median(x: np.ndarray, win_samples: int) -> np.ndarray:
    w = int(max(1, win_samples))
    if w == 1 or x.size < w:
        return x.copy()
    sw = sliding_window_view(x, w)
    med = np.median(sw, axis=-1)
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    return np.pad(med, (pad_left, pad_right), mode="edge")

def _rolling_quantile(x: np.ndarray, win_samples: int, quantile: float) -> np.ndarray:
    """Скользящий квантиль для расчета базального тонуса матки."""
    w = int(max(1, win_samples))
    if w == 1 or x.size < w:
        return x.copy()
    sw = sliding_window_view(x, w)
    quant = np.quantile(sw, quantile, axis=-1)
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    return np.pad(quant, (pad_left, pad_right), mode="edge")

def _moving_median(x: np.ndarray, win_samples: int) -> np.ndarray:
    """Алиас для _rolling_median для совместимости."""
    return _rolling_median(x, win_samples)

def _hampel(y: np.ndarray, win_samples: int = 21, n_sigmas: float = 3.0) -> np.ndarray:
    w = int(max(3, win_samples | 1))  # нечётное
    if y.size < w:
        return y.copy()
    sw = sliding_window_view(y, w)
    med = np.median(sw, axis=-1)
    mad = np.median(np.abs(sw - med[..., None]), axis=-1)
    sigma = 1.4826 * mad
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    med = np.pad(med, (pad_left, pad_right), mode="edge")
    sigma = np.pad(sigma, (pad_left, pad_right), mode="edge")
    out = y.copy()
    mask = np.abs(out - med) > n_sigmas * (sigma + 1e-9)
    out[mask] = med[mask]
    return out

def _interp_nans(x: np.ndarray) -> np.ndarray:
    y = x.astype(float).copy()
    n = len(y)
    if n == 0:
        return y
    mask = np.isnan(y)
    if not mask.any():
        return y
    good = ~mask
    if good.sum() == 0:
        raise ValueError("Весь ряд состоит из NaN.")
    idx = np.arange(n)
    y[mask] = np.interp(idx[mask], idx[good], y[good])
    return y

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
        if s - pe <= max_gap:     # сливаем через короткий разрыв
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
        if kind == "accel":
            peak = np.max(seg)
        else:
            peak = -np.min(seg)
        if peak >= min_peak:
            out.append((s, e))
    return out

def _hysteresis_mask(delta: np.ndarray, enter_thr: float, exit_thr: float, sign: int) -> np.ndarray:
    """sign=+1 для акцелераций, -1 для децелераций."""
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




def classify_and_plot_fhr(
    fhr_bpm: np.ndarray,
    fs: float = 7.87,
    class_name: str = '',
    *,
    # — базовая классификация по устойчивой базе (как было)
    brady_threshold: float = 120.0,
    tachy_threshold: float = 160.0,
    severe_brady_threshold: float = 100.0,
    severe_tachy_threshold: float = 180.0,
    sustain_seconds: float = 600.0,
    smooth_seconds: float = 5.0,
    chunk_minutes: float = 20.0,
    show: bool = True,
    # — улучшенная детекция событий
    event_baseline_seconds: float = 90.0,   # локальная база (скользящая медиана)
    hampel_seconds: float = 3.0,            # подавление выбросов
    accel_enter: float = 12.0, accel_exit: float = 8.0,
    decel_enter: float = 12.0, decel_exit: float = 8.0,
    accel_min_seconds: float = 15.0, decel_min_seconds: float = 15.0,
    accel_min_peak: float = 15.0, decel_min_depth: float = 15.0,
    merge_gap_seconds: float = 5.0,
    prolonged_decel_seconds: float = 120.0,
    very_prolonged_decel_seconds: float = 300.0,
    bg_alpha: float = 0.28,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:

    x = np.asarray(fhr_bpm, float).ravel()
    if x.size == 0:
        raise ValueError("Пустой временной ряд ЧСС.")
    x = _interp_nans(x)

    # мягкое сглаживание + удаление выбросов
    smooth_win = max(1, int(round(fs * smooth_seconds)))
    x_smooth = _moving_avg(x, smooth_win)
    x_smooth = _hampel(x_smooth, max(3, int(fs * hampel_seconds) | 1))

    # базальная линия для КЛАССИФИКАЦИИ (длинная)
    sustain_win = max(1, int(round(fs * sustain_seconds)))
    baseline_state = _moving_avg(x_smooth, sustain_win)

    # >>> ПРЕДОХРАНИТЕЛЬ от рассинхрона <<<
    baseline_state = _resample_to_len(baseline_state, x.size)

    # состояния
    states = np.full_like(x, 1, dtype=int)
    states[baseline_state < brady_threshold] = 2
    states[baseline_state < severe_brady_threshold] = 3
    states[baseline_state > tachy_threshold] = 4
    states[baseline_state > severe_tachy_threshold] = 5

    # базальная линия для СОБЫТИЙ (локальная, робастная)
    ev_win = max(1, int(round(fs * event_baseline_seconds)))
    baseline_event = _rolling_median(x_smooth, ev_win)

    # >>> и тут — для симметрии/надёжности <<<
    baseline_event = _resample_to_len(baseline_event, x.size)
    delta = x_smooth - baseline_event

    # гистерезисные маски
    accel_mask0 = _hysteresis_mask(delta, accel_enter, accel_exit, sign=+1)
    decel_mask0 = _hysteresis_mask(delta, decel_enter, decel_exit, sign=-1)

    # слияние через короткие разрывы
    merge_gap = int(round(fs * merge_gap_seconds))
    accel_segs = _merge_segments(_segments_from_mask(accel_mask0), merge_gap)
    decel_segs = _merge_segments(_segments_from_mask(decel_mask0), merge_gap)

    # финальная фильтрация по длительности и реальному пику
    min_len_acc = int(round(fs * accel_min_seconds))
    min_len_dec = int(round(fs * decel_min_seconds))
    accel_segs = _filter_by_len_and_peak(delta, accel_segs, min_len_acc, accel_min_peak, "accel")
    decel_segs = _filter_by_len_and_peak(delta, decel_segs, min_len_dec, decel_min_depth, "decel")

    # длительные децелерации
    durs_dec = [(e - s) / fs for s, e in decel_segs]
    decel_long = [seg for seg, d in zip(decel_segs, durs_dec) if d > prolonged_decel_seconds]
    decel_very_long = [seg for seg, d in zip(decel_segs, durs_dec) if d >= very_prolonged_decel_seconds]

    # статистика
    n = x.size
    statistics: Dict[str, float] = {STATE_LABELS[s]: float(np.sum(states == s) / fs) for s in STATE_LABELS}
    statistics["Всего, сек"] = float(n / fs)

    cp = np.flatnonzero(np.diff(states)) + 1
    seg_starts = np.r_[0, cp]; seg_states = states[seg_starts]
    for s in (2, 3, 4, 5):
        statistics[f"Промежутки (шт) — {STATE_LABELS[s]}"] = float(int(np.sum(seg_states == s)))
    statistics["Промежутки (шт) — Не норма (всего)"] = float(int(np.sum(seg_states != 1)))

    def _dur(segs): return sum((e - s) for s, e in segs) / fs
    def _amps(segs, kind):
        vals = []
        for s, e in segs:
            seg = delta[s:e]
            vals.append(float(np.max(seg) if kind == "accel" else -np.min(seg)))
        return vals





    acc_durs = [(e - s) / fs for s, e in accel_segs]
    dec_durs = [(e - s) / fs for s, e in decel_segs]
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

        "Децелерации — пролонгированные >2 мин (шт)": float(len(decel_long)),
        "Децелерации — ≥5 мин (шт)": float(len(decel_very_long)),
    })

    # график
    if show:
        max_chunk = max(1, int(round(chunk_minutes * 60 * fs)))
        t_abs_min = np.arange(n) / fs / 60.0
        for chunk_idx, cs in enumerate(range(0, n, max_chunk)):
            ce = min(n, cs + max_chunk)
            t = (np.arange(cs, ce) - cs) / fs / 60.0 + chunk_minutes * chunk_idx

            fig, ax = plt.subplots(figsize=(12, 5))

            # фоновые спаны (поверх осевого фона)
            def _draw(segs, color, alpha):
                for s0, e0 in segs:
                    s = max(s0, cs); e = min(e0, ce)
                    if e <= s: continue
                    t0 = (s - cs) / fs / 60.0 + chunk_minutes * chunk_idx
                    t1 = (e - cs) / fs / 60.0 + chunk_minutes * chunk_idx
                    ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=1)

            _draw(accel_segs, BG_ACCEL, bg_alpha)
            _draw(decel_segs, BG_DECEL, bg_alpha * 0.95)
            _draw(decel_long, BG_DECEL_LONG, min(bg_alpha + 0.05, 0.35))

            # цветная линия по состояниям
            change_points = np.flatnonzero(np.diff(states)) + 1
            bounds = [cs] + [p for p in change_points if cs < p < ce] + [ce]
            shown = set()
            for a, b in zip(bounds[:-1], bounds[1:]):
                st = states[a]; lab = None
                if st != 1 and st not in shown:
                    lab = STATE_LABELS[st]; shown.add(st)
                ax.plot(t[(a - cs):(b - cs)], x[a:b], lw=1.8, color=STATE_COLORS[st], label=lab, zorder=2)

            # базальная линия (для событий — более наглядная)
            ax.plot(t, baseline_event[cs:ce], lw=1.0, ls="--", color="#7f8c8d",
                    label=f"Базальная (локальная {int(event_baseline_seconds)} сек)", zorder=3)

            for thr in (severe_brady_threshold, brady_threshold, tachy_threshold, severe_tachy_threshold):
                ax.axhline(thr, ls=":", lw=1.0, color="#7f8c8d")

            proxies = [
                Patch(facecolor=BG_ACCEL, alpha=bg_alpha, label="Акцелерации"),
                Patch(facecolor=BG_DECEL, alpha=bg_alpha*0.95, label="Децелерации"),
                # Patch(facecolor=BG_DECEL_LONG, alpha=min(bg_alpha+0.05, 0.35), label="Децелерации >2 мин"),
            ]

            chunk_start = t_abs_min[cs]
            chunk_end = t_abs_min[ce-1] if ce > cs else chunk_start
            ax.set_title(f"ЧСС с устойчивостью ≥ {int(sustain_seconds)} сек. Диагноз: {class_name}")
            ax.set_xlabel("Время, мин"); ax.set_ylabel("ЧСС, уд/мин")
            leg1 = ax.legend(loc="upper right"); ax.add_artist(leg1)
            ax.legend(handles=proxies, loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.show()

    return states, baseline_event, statistics


def analyze_and_plot_toco_demo(
    toco_mmHg: np.ndarray,
    fs: float,
    *,
    show: bool = True,
    # сглаживание/тонус
    smooth_seconds: float = 3.0,           # 3–5 с
    smooth_kind: str = "mean",             # "mean" | "median"
    tone_window_seconds: float = 75.0,     # 60–90 с
    tone_quantile: float = 0.10,           # 10%
    # схватки
    amp_threshold_mmHg: float = 10.0,      # давление − тонус ≥ 10
    min_duration_seconds: float = 30.0,    # минимум 30 с
    end_below_seconds: float = 12.0,       # конец: ниже порога ≥10–15 с
    # окна тревог
    win_len_seconds: float = 600.0,        # 10 минут
    win_step_seconds: float = 60.0,        # шаг 60 с
    # отрисовка
    chunk_minutes: float = 20.0,
    alpha_bg: float = 0.28,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], List[Dict], Dict[str, float]]:
    """
    DEMO: фон = только тревоги (тахисистолия/гипертонус), плашки = тетанические,
    ЛИНИЯ окрашена по интенсивности (3 уровня).

    Возвращает:
      tone               — базальный тонус (мм рт. ст.), shape=(n,)
      intensity_mmHg     — интенсивность схватки (excess>=0), мм рт. ст., shape=(n,)
      alert_code         — код тревоги по времени: 0 ок, 1 тахисистолия, 2 гипертонус, 3 тетания, shape=(n,)
      contractions       — список детектированных схваток
      windows_alerts     — список окон тревог
      stats_alerts       — сводка
    """
    x = _interp_nans(toco_mmHg)
    n = x.size
    if n == 0:
        raise ValueError("Пустая токограмма.")

    # сглаживание
    w_sm = max(1, int(round(fs * smooth_seconds)))
    xs = _moving_median(x, w_sm) if smooth_kind == "median" else _moving_avg(x, w_sm)

    # базальный тонус: нижний «конверт» (q10)
    w_tone = max(1, int(round(fs * tone_window_seconds)))
    tone = _rolling_quantile(xs, w_tone, tone_quantile)

    # превышение над тоном и непрерывная интенсивность
    excess = xs - tone
    intensity_mmHg = np.maximum(excess, 0.0)

    # детекция схваток (с хвостом завершения)
    thr = float(amp_threshold_mmHg)
    min_len = int(round(fs * min_duration_seconds))
    gap_close = int(round(fs * end_below_seconds))

    raw = _segments_from_mask(excess >= thr)
    merged = []
    if raw:
        cs, ce = raw[0]
        for s, e in raw[1:]:
            if s - ce <= gap_close:
                ce = e
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
    segs = [(s, e) for s, e in merged if (e - s) >= min_len]

    contractions: List[Dict] = []
    peaks_idx = []
    for s, e in segs:
        seg_exc = excess[s:e]
        if seg_exc.size == 0:
            continue
        loc = int(np.argmax(seg_exc))
        p_idx = s + loc
        peaks_idx.append(p_idx)
        contractions.append({
            "start_s": s / fs,
            "end_s": e / fs,
            "duration_s": (e - s) / fs,
            "amp_mmHg": float(seg_exc[loc]),
            "peak_s": p_idx / fs,
            "peak_idx": int(p_idx),
        })
    peaks_idx = np.asarray(peaks_idx, int)

    # окна тревог (только тахисистолия/гипертонус)
    L = int(round(fs * win_len_seconds))
    S = int(round(fs * win_step_seconds))
    starts = list(range(0, max(1, n - L + 1), S)) or [0]

    COLORS = {
        "Тахисистолия": "#e74c3c",
        "Гипертонус":   "#6e2c00",
        "Тетания":      "#8e44ad",  # плашки схваток ≥120 с
    }

    windows_alerts: List[Dict] = []
    for s0 in starts:
        e0 = min(n, s0 + L)
        if peaks_idx.size:
            n_cnt = int(np.count_nonzero((peaks_idx >= s0) & (peaks_idx < e0)))
        else:
            n_cnt = 0
        tone_med = float(np.median(tone[s0:e0])) if e0 > s0 else 0.0

        label, color, hatch = None, None, None
        if n_cnt > 5:
            label, color = "Тахисистолия", COLORS["Тахисистолия"]
        elif tone_med > 25.0:
            label, color, hatch = "Гипертонус", COLORS["Гипертонус"], "////"




        windows_alerts.append({
            "start_s": s0 / fs, "end_s": e0 / fs,
            "label": label, "color": color, "hatch": hatch,
            "n": n_cnt, "tone_med": tone_med,
        })

    # сводка тревог
    total_sec = n / fs
    n_win = len(windows_alerts) if windows_alerts else 1
    tachy_cnt = sum(1 for w in windows_alerts if w["label"] == "Тахисистолия")
    hyper_cnt = sum(1 for w in windows_alerts if w["label"] == "Гипертонус")
    tetanic = [c for c in contractions if c["duration_s"] >= 120.0]

    stats_alerts: Dict[str, float] = {
        "Длительность записи, мин": float(total_sec / 60.0),
        "Тахисистолия — окна (шт)": float(tachy_cnt),
        "Тахисистолия — окна (%)": float(100.0 * tachy_cnt / n_win),
        "Гипертонус — окна (шт)": float(hyper_cnt),
        "Гипертонус — окна (%)": float(100.0 * hyper_cnt / n_win),
        "Тетанические — эпизоды (шт)": float(len(tetanic)),
        "Тетанические — макс длительность, с": float(max([c["duration_s"] for c in tetanic]) if tetanic else 0.0),
    }

    # --- формирование ряда alert_code по каждому сэмплу ---
    ALERT_CODE = {"Тахисистолия": 1, "Гипертонус": 2, "Тетания": 3}
    alert_code = np.zeros(n, dtype=np.int8)

    # 1) тетания (наивысший приоритет)
    for c in tetanic:
        s = max(0, int(round(c["start_s"] * fs)))
        e = min(n, int(round(c["end_s"]   * fs)))
        if e > s:
            alert_code[s:e] = np.maximum(alert_code[s:e], ALERT_CODE["Тетания"])

    # 2) окна тахисистолии/гипертонуса
    for w in windows_alerts:
        if w["label"] is None:
            continue
        s = max(0, int(round(w["start_s"] * fs)))
        e = min(n, int(round(w["end_s"]   * fs)))
        code = ALERT_CODE[w["label"]]
        if e > s:
            alert_code[s:e] = np.maximum(alert_code[s:e], code)

    # --- отрисовка (без изменений логики) ---
    if show:
        LINE_COLORS = {0: "#555555", 1: "#3498db", 2: "#27AE60", 3: "#1abc9c"}
        t_abs_min = np.arange(n) / fs / 60.0
        chunk_samples = int(round(fs * 60.0 * chunk_minutes))

        for cs in range(0, n, chunk_samples):
            ce = min(n, cs + chunk_samples)
            t = t_abs_min[cs:ce]
            fig, ax = plt.subplots(figsize=(12, 4.8))

            frag_s, frag_e = cs / fs, ce / fs
            for w in windows_alerts:
                if w["label"] is None:
                    continue
                w0, w1 = w["start_s"], w["end_s"]
                ov0 = max(w0, frag_s)
                ov1 = min(w1, frag_e)
                if ov1 <= ov0:
                    continue
                span = ax.axvspan(ov0/60.0, ov1/60.0, facecolor=w["color"], alpha=alpha_bg, zorder=0)
                if w.get("hatch"):
                    span.set_hatch(w["hatch"])

            for c in contractions:
                if c["duration_s"] >= 120.0:
                    s_t = max(c["start_s"], frag_s)
                    e_t = min(c["end_s"],   frag_e)
                    if e_t > s_t:
                        ax.axvspan(s_t/60.0, e_t/60.0, facecolor=COLORS["Тетания"], alpha=0.25, zorder=1)

            exc_seg = excess[cs:ce]
            band = np.zeros(ce - cs, dtype=np.int8)
            band[(exc_seg >= 10.0) & (exc_seg < 30.0)] = 1
            band[(exc_seg >= 30.0) & (exc_seg < 80.0)] = 2
            band[(exc_seg >= 80.0)]                     = 3
            change = np.flatnonzero(np.diff(band)) + 1
            bounds = [0] + change.tolist() + [ce - cs]
            for a_rel, b_rel in zip(bounds[:-1], bounds[1:]):
                key = int(band[a_rel])
                ax.plot(t[a_rel:b_rel], xs[cs+a_rel:cs+b_rel], lw=1.8, color=LINE_COLORS[key], zorder=2)

            ax.plot(t, tone[cs:ce], lw=1.0, ls="--", color="#9e9e9e", zorder=3)




            proxies = [
                Patch(facecolor=COLORS["Тахисистолия"], alpha=alpha_bg, label="Тахисистолия (>5/10 мин)"),
                Patch(facecolor=COLORS["Гипертонус"],   alpha=alpha_bg, label="Гипертонус (tone_med > 25)", hatch="////"),
                Patch(facecolor=COLORS["Тетания"],      alpha=0.25,       label="Тетаническая (≥120 с)"),
                Line2D([0],[0], color=LINE_COLORS[1], lw=2, label="Интенсивность 10–30"),
                Line2D([0],[0], color=LINE_COLORS[2], lw=2, label="Интенсивность 30–80"),
                Line2D([0],[0], color=LINE_COLORS[3], lw=2, label="Интенсивность ≥80"),
            ]
            ax.legend(handles=proxies, loc="upper left", ncol=2, framealpha=0.95)

            ax.set_title(f"Токограмма")
            ax.set_xlabel("Время, мин"); ax.set_ylabel("Маточные сокращения, мм рт. ст.")
            ax.grid(True, alpha=0.25)
            plt.show()

    return tone, intensity_mmHg, alert_code, contractions, windows_alerts, stats_alerts
