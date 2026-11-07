import numpy as np


def _gap_spans(missing):
    spans = []
    indices = np.flatnonzero(missing)
    if indices.size == 0:
        return spans
    start = int(indices[0])
    previous = int(indices[0])
    for current in indices[1:]:
        current = int(current)
        if current == previous + 1:
            previous = current
            continue
        spans.append((start, previous + 1))
        start = current
        previous = current
    spans.append((start, previous + 1))
    return spans


def _fill_nearest(values, start, end):
    length = len(values)
    left = start - 1
    while left >= 0 and not np.isfinite(values[left]):
        left -= 1
    right = end
    while right < length and not np.isfinite(values[right]):
        right += 1
    for pos in range(start, end):
        if left >= 0 and right < length:
            if (pos - left) <= (right - pos):
                values[pos] = values[left]
            else:
                values[pos] = values[right]
        elif left >= 0:
            values[pos] = values[left]
        elif right < length:
            values[pos] = values[right]


def _fill_diurnal(values, start, end, period, diurnal_days):
    for pos in range(start, end):
        collected = []
        for day in range(1, diurnal_days + 1):
            source = pos - day * period
            if source >= 0 and np.isfinite(values[source]):
                collected.append(values[source])
        if collected:
            values[pos] = float(np.mean(collected))
    if np.isnan(values[start:end]).any():
        _fill_nearest(values, start, end)


def _fill_linear(values, start, end, period, diurnal_days):
    left = start - 1
    right = end
    length = len(values)
    if left >= 0 and right < length and np.isfinite(values[left]) and np.isfinite(values[right]):
        span = right - left
        for pos in range(start, end):
            weight = (pos - left) / span
            values[pos] = values[left] + weight * (values[right] - values[left])
        return
    _fill_diurnal(values, start, end, period, diurnal_days)


def repair_series(series, period, short_gap_limit, diurnal_days):
    values = np.array(series, dtype=np.float64, copy=True)
    missing = ~np.isfinite(values)
    if not np.any(missing):
        return values
    for start, end in _gap_spans(missing):
        if (end - start) > short_gap_limit:
            _fill_diurnal(values, start, end, period, diurnal_days)
        else:
            _fill_linear(values, start, end, period, diurnal_days)
    return values


def rolling_median(series, window):
    half = window // 2
    length = len(series)
    output = np.empty(length, dtype=np.float64)
    for index in range(length):
        left = max(0, index - half)
        right = min(length, index + half + 1)
        output[index] = float(np.median(series[left:right]))
    return output


def suppress_outliers(series, threshold, window):
    values = np.array(series, dtype=np.float64, copy=True)
    center = float(np.mean(values))
    spread = float(np.std(values))
    if spread < 1e-8:
        return values
    flagged = np.abs(values - center) > threshold * spread
    if not np.any(flagged):
        return values
    median = rolling_median(values, window)
    values[flagged] = median[flagged]
    return values


def repair_window(window, period, short_gap_limit, diurnal_days, outlier_lambda, rolling_window):
    repaired = np.empty_like(window, dtype=np.float64)
    for channel in range(window.shape[1]):
        filled = repair_series(window[:, channel], period, short_gap_limit, diurnal_days)
        repaired[:, channel] = suppress_outliers(filled, outlier_lambda, rolling_window)
    if not np.isfinite(repaired).all():
        raise RuntimeError("sensor window still contains missing values after repair")
    return repaired.astype(np.float32)


def repair_windows(windows, cfg):
    minutes = cfg["preprocess"]["interval_minutes"]
    period = int(round(24 * 60 / minutes))
    short_gap_limit = int(round(cfg["preprocess"]["gap_hours"] * 60 / minutes))
    repaired = np.empty_like(windows, dtype=np.float32)
    for index in range(windows.shape[0]):
        repaired[index] = repair_window(
            windows[index],
            period,
            short_gap_limit,
            cfg["preprocess"]["diurnal_days"],
            cfg["preprocess"]["outlier_lambda"],
            cfg["preprocess"]["rolling_median"],
        )
    return repaired


def channel_scaler(array):
    flat = array.reshape(-1, array.shape[-1]).astype(np.float64)
    center = flat.mean(axis=0)
    spread = flat.std(axis=0)
    spread = np.where(spread < 1e-6, 1.0, spread)
    return center.astype(np.float32), spread.astype(np.float32)


def apply_scaler(array, center, spread):
    return ((array - center) / spread).astype(np.float32)


def verify_isolated_gap_interpolation():
    probe = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    filled = repair_series(probe, period=48, short_gap_limit=6, diurnal_days=3)
    if abs(float(filled[1]) - 2.0) > 1e-8:
        raise RuntimeError("isolated gap interpolation does not match the manuscript formula")
