"""Accuracy check: time-based hold-out, no future data in the context."""

import numpy as np

from app.ai_core import predict_delays
from app.config import EXCLUDED, FEATURES, RPT_CONTEXT_ROWS, TARGET
from app.history import load_history


def evaluate(test_rows=256):
    history = load_history()
    cut = int(len(history) * 0.8)
    past, future = history.iloc[:cut], history.iloc[cut:]
    test = future.sample(min(test_rows, len(future)), random_state=42)
    y_true = test[TARGET].to_numpy(float)
    y_pred = predict_delays(past, test)

    def scores(pred):
        err = y_true - pred
        return {"MAE": round(float(np.abs(err).mean()), 3),
                "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
                "R2": round(float(1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()), 3)}

    return {
        "TestRows": len(test),
        "ContextRowsPerCall": min(RPT_CONTEXT_ROWS, len(past)),
        "Split": "train = oldest 80% of POs, test = sample of newest 20%",
        "SAP_RPT_1": scores(y_pred),
        "Baseline_AverageDelay": scores(np.full_like(y_true, past[TARGET].mean())),
        "Features": FEATURES,
        "Excluded": EXCLUDED,
    }
