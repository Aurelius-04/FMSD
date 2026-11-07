import numpy as np


def regression_metrics(y_true, y_pred):
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    error = pred - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denom = np.where(np.abs(truth) < 1e-8, 1e-8, truth)
    mape = float(100.0 * np.mean(np.abs(error / denom)))
    residual = float(np.sum(np.square(error)))
    centered = float(np.sum(np.square(truth - np.mean(truth))))
    r2 = 0.0 if centered < 1e-12 else float(1.0 - residual / centered)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def cycle_metrics(y_true, y_pred, cycle_ids):
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    cycles = np.asarray(cycle_ids)
    predicted = []
    observed = []
    for cycle_id in np.unique(cycles):
        mask = cycles == cycle_id
        target = truth[mask]
        if np.max(np.abs(target - target[0])) > 1e-4:
            raise RuntimeError("a compartment-season does not share one net-return target")
        predicted.append(float(np.mean(pred[mask])))
        observed.append(float(target[0]))
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - observed
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    residual = float(np.sum(np.square(error)))
    centered = float(np.sum(np.square(observed - np.mean(observed))))
    r2 = 0.0 if centered < 1e-12 else float(1.0 - residual / centered)
    return {"mae": mae, "rmse": rmse, "r2": r2, "cycles": int(observed.shape[0])}
