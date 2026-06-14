"""Entrenamiento del modelo 1X2 (local gana / empate / visitante).

Mejores prácticas aplicadas:
  - VALIDACIÓN TEMPORAL walk-forward (TimeSeriesSplit), no un solo corte: estima el
    rendimiento de forma más fiable entrenando con el pasado y validando con el futuro,
    repetido en varios tramos.
  - TUNING de hiperparámetros por esa validación temporal.
  - CALIBRACIÓN de probabilidades (isotónica): que un "60%" ocurra ~60% de las veces.
  - CHAMPION/CHALLENGER: el modelo nuevo solo reemplaza al vigente si mejora el log-loss.

Dependencias de ML opcionales: `uv sync --extra ml`.
"""

from __future__ import annotations

from pathlib import Path

from worldcup.common.db import connect

_FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "rest_days_home", "rest_days_away",
    "home_form_pts", "away_form_pts", "form_pts_diff",
    "home_gf", "home_ga", "away_gf", "away_ga",
    "h2h_home_ppg",
    "is_neutral", "importance",        # sede neutral + importancia de competición
    "home_sos", "away_sos",            # strength-of-schedule
]
_MODEL_PATH = Path("models/model_1x2.joblib")

# Rejilla pequeña de hiperparámetros (suficiente para tabular; mantiene el runtime acotado).
_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400},
    {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 500},
    {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 200},
]


def _make_xgb(params: dict):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        subsample=0.8, colsample_bytree=0.8, **params,
    )


def train_1x2(test_frac: float = 0.2) -> dict:
    """Entrena el clasificador 1X2 con tuning + calibración + validación temporal."""
    import joblib
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import TimeSeriesSplit

    with connect() as con:
        df = con.sql("SELECT * FROM match_features ORDER BY kickoff_utc").pl()
    if df.height < 300:
        raise ValueError(f"Solo {df.height} partidos con features; ingiere más datos.")

    cut = int(df.height * (1 - test_frac))
    train, test = df[:cut], df[cut:]
    X_tr = train.select(_FEATURES).to_numpy()
    y_tr = train["result_1x2"].to_numpy()
    X_te = test.select(_FEATURES).to_numpy()
    y_te = test["result_1x2"].to_numpy()

    # --- Tuning por validación temporal (walk-forward) sobre el TRAIN ---
    tscv = TimeSeriesSplit(n_splits=4)
    best, best_ll = None, float("inf")
    for params in _GRID:
        fold_lls = []
        for tr_idx, va_idx in tscv.split(X_tr):
            m = _make_xgb(params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            p = m.predict_proba(X_tr[va_idx])
            fold_lls.append(log_loss(y_tr[va_idx], p, labels=[0, 1, 2]))
        mean_ll = float(np.mean(fold_lls))
        if mean_ll < best_ll:
            best_ll, best = mean_ll, params

    # --- Modelo final CALIBRADO (isotónico, con folds temporales) ---
    calibrated = CalibratedClassifierCV(
        _make_xgb(best), method="isotonic", cv=TimeSeriesSplit(n_splits=3)
    )
    calibrated.fit(X_tr, y_tr)

    proba = calibrated.predict_proba(X_te)
    preds = proba.argmax(axis=1)
    metrics = {
        "n_train": int(train.height), "n_test": int(test.height),
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "baseline_accuracy_siempre_local": round(float((y_te == 0).mean()), 4),
        "log_loss": round(float(log_loss(y_te, proba, labels=[0, 1, 2])), 4),
        "cv_log_loss_tuning": round(best_ll, 4),
        "best_params": best,
        "calibrado": True,
    }

    # --- Champion/challenger: promociona solo si mejora el log-loss (menor = mejor) ---
    from worldcup.predictor.registry import promote

    def _save():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": calibrated, "features": _FEATURES}, _MODEL_PATH)

    decision = promote("1x2", metrics["log_loss"], _save,
                       higher_is_better=False, metrics=metrics)
    metrics["promoted"] = decision["promoted"]
    metrics["previous_log_loss"] = decision["previous_score"]

    from worldcup.tracking import track
    track("wc-1x2", {"model": "XGB+isotonic", "best_params": str(best)}, metrics)
    return metrics
